import conf
import fedml
import os

import multiprocessing
import pandas as pd

from autoscaler.autoscaler_db import AutoscalerDiskDB, AutoscalerKVCache
from utils import logger
from utils.singleton import Singleton

from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache

# We need to keep the state in a global variable. We shall also use locks 
# when accessing the state map to avoid deadlocks / race conditions.
# AUTOSCALER_DISK_DB = None
# AUTOSCALER_KV_CACHE = None
# AUTOSCALER_KV_CACHE_KEY_FORMAT = "{}"
# AUTOSCALER_LOCK = Lock()

class FedMLAutoscaler(multiprocessing.Process, metaclass=Singleton):

    def __init__(self, redis_addr, redis_port, redis_password):        
        self.redis_addr = multiprocessing.Value("redis_addr", redis_addr)
        self.redis_port = multiprocessing.Value("redis_port", redis_port)
        self.redis_password = multiprocessing.Value("redis_password", redis_password)
            
        # """ Initialize autoscaler's database """
        # global AUTOSCALER_DISK_DB, AUTOSCALER_KV_CACHE
        # if AUTOSCALER_DISK_DB is None \
        #     and AUTOSCALER_KV_CACHE is None:
        #     self.__initializedb()        

    def run(self) -> None:        
        FedMLModelCache.get_instance()

    def __initializedb(self):
        global \
            AUTOSCALER_DISK_DB, \
            AUTOSCALER_KV_CACHE, \
            AUTOSCALER_KV_CACHE_KEY_FORMAT
        AUTOSCALER_DISK_DB = AutoscalerDiskDB.get_instance(
            dbpath=conf.AUTOSCALER_DB_SQLITE_PATH)        
        AUTOSCALER_KV_CACHE = AutoscalerKVCache.get_instance(
            values_per_key=conf.AUTOSCALER_VALUE_CACHE_SIZE)
        logger.info("Fetching data from the previous on-disk state.")
        persisted_data = AUTOSCALER_DISK_DB.select_query_data(
            records_per_query=conf.AUTOSCALER_VALUE_CACHE_SIZE)
        logger.info("Fetched data from disk.")
        logger.info("Formatting persisted data for in-memory KV cache.")
        persisted_data_formatted = []
        for record in persisted_data:
            cache_key = AUTOSCALER_KV_CACHE_KEY_FORMAT.format(
                record.pop("endpoint_id"))
            # The `insert` function expects 
            # the record to be a list type.
            record = [record]
            persisted_data_formatted.append({cache_key: record})
        logger.info("Formatted persisted data for in-memory KV cache.")
        logger.info("Populating in-memory database state.")
        AUTOSCALER_KV_CACHE.bulk_insert(
                persisted_data_formatted)
        logger.info("Populated in-memory database state. Total records: {}"\
                    .format(len(persisted_data_formatted)))

    @staticmethod
    def get_instance():
        return FedMLAutoscaler()

    @staticmethod
    def set_config_version():
        current_env_version = os.getenv(conf.CONFIG_VERSION_NAME,
                                        conf.CONFIG_VERSION_DEFAULT)
        fedml.set_env_version(current_env_version)

    def reactive(self, request_endpoint_id, request_total_requests, request_data):

        global \
            AUTOSCALER_DISK_DB, \
            AUTOSCALER_KV_CACHE, \
            AUTOSCALER_KV_CACHE_KEY_FORMAT, \
            AUTOSCALER_LOCK
        allocate_instance = 0
        free_instance = 0
        trigger_predictive = 0
        error_code, error_message = -1, ""
            
        try:
            FedMLAutoscaler.set_config_version()
            cache_key = AUTOSCALER_KV_CACHE_KEY_FORMAT.format(request_endpoint_id)
            # Compute within the lock context.
            with AUTOSCALER_LOCK:
                # Example State Map Format:
                #   {
                #        '5': [{'timestamp': '2024-01-26T13:57:13z', 'latency': 4.731, 'qps': 90.946}]
                #        '8': [
                #               {'timestamp': '2024-01-26T13:23:13z', 'latency': 1.615, 'qps': 43.458},
                #               {'timestamp': '2024-01-26T13:23:13z', 'latency': 1.615, 'qps': 43.458},
                #             ]                
                #   }
                # Better to append the new data points and then sort.
                AUTOSCALER_KV_CACHE.insert(cache_key, request_data)
                cached_data = AUTOSCALER_KV_CACHE.get_by_key(cache_key, sorted=True)

                df = pd.DataFrame.from_records(cached_data)
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index, format=conf.CONFIG_DATETIME_FORMAT)                
                
                # for every rolling operation we get the 
                # last value from the computed series, i.e., [-1]
                latency_5min = round(df['latency'].rolling('5min').mean()[-1], 3)
                latency_15min = round(df['latency'].rolling('15min').mean()[-1], 3)
                latency_30min = round(df['latency'].rolling('30min').mean()[-1], 3)
                latency_60min = round(df['latency'].rolling('60min').mean()[-1], 3)
                
                qps_5min = round(df['qps'].rolling('5min').mean()[-1], 3)
                qps_15min = round(df['qps'].rolling('15min').mean()[-1], 3)
                qps_30min = round(df['qps'].rolling('30min').mean()[-1], 3)
                qps_60min = round(df['qps'].rolling('60min').mean()[-1], 3)

                # ALLOCATE RESOURCES CRITERION: 
                #   If the 5-min QPS SMA is smaller than the 15-min QPS SMA 
                #   we need to increase throughput (high traffic).
                # FREE RESOURCES CRITERION: 
                #   If the 30-min Latency SMA is greater than the 5-min Latency SMA 
                #   we need to release resources (low traffic).
                # PREDICTIVE AUTOSCALING CRITERION:
                #   If both allocate and free instance variables are true then we 
                #   should trigger a more sophisticated ML/DL predictive approach.
                # TODO(fedml-dimitris): account for NaN values.
                # TODO(fedml-dimitris): can we return the number of instances we 
                #                       need to acquire and release?
                if qps_5min < qps_30min:
                    allocate_instance = 1 # allocate one instance
                if latency_15min > latency_30min:
                    free_instance = 1 # release one instance
                if allocate_instance and free_instance:
                    allocate_instance = 0
                    free_instance = 0
                    trigger_predictive = 1
                
                # Finally populate the on disk (sqlite) db with 
                # the data records of the incoming request.
                AUTOSCALER_DISK_DB.populate_db(
                    request_endpoint_id,
                    request_total_requests,
                    "reactive",                    
                    allocate_instance,
                    free_instance, 
                    trigger_predictive, 
                    request_data)

        except Exception as e:
            logger.exception(e)
            error_code = fedml.api.constants.ApiConstants.ERROR_CODE[fedml.api.constants.ApiConstants.ERROR]
            error_message = f"{str(e)}"

        return allocate_instance, free_instance, trigger_predictive, error_code, error_message

    def predictive(self, request_endpoint_id, request_total_requests, request_data):
        error_code = fedml.api.constants.ApiConstants.ERROR_CODE[fedml.api.constants.ApiConstants.ERROR]
        error_message = "Not yet implemented."
        return None, None, None, error_code, error_message
