import sys
sys.path.insert(0, '..') # Need to extend the path because the test script is a standalone script. 

import common as common
import copy
import json
import random
import time

import numpy as np

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import logger

random.seed(0)


class StressTest(object):    

    STRESS_TEST_STRUCTURE = namedtuple('stress_test', ['endpoint_id', 'payloads', 'responses'])    
    WARMUP_ENDPOINTS_ENV_TEST_STATS = common.endpoints_stats(
        "data/nexusai_endpoint_env_test.csv")
    WARMUP_ENDPOINTS_ENV_PROD_STATS = common.endpoints_stats(
        "data/nexusai_endpoint_env_prod.csv")    

    @classmethod
    def warmup(cls, num_values):
        # We will select values randonly within the IQR (Inter Quartile Range) 
        # range of distribution IQR = [Q3-Q1]  Q1 | Median | Q3
        # to avoid small and large outliers both for qps and latency
        qps_q1 = np.percentile(cls.WARMUP_ENDPOINTS_ENV_TEST_STATS.all_qps, 25)
        qps_q3 = np.percentile(cls.WARMUP_ENDPOINTS_ENV_TEST_STATS.all_qps, 75)
        qps_values = np.round(np.random.uniform(qps_q1, qps_q3, num_values), 3)
        qps_values = qps_values.tolist()

        latency_q1 = np.percentile(cls.WARMUP_ENDPOINTS_ENV_TEST_STATS.all_latency, 25)
        latency_q3 = np.percentile(cls.WARMUP_ENDPOINTS_ENV_TEST_STATS.all_latency, 75)
        latency_values = np.round(np.random.uniform(latency_q1, latency_q3, num_values), 3)
        latency_values = latency_values.tolist()

        # We return a list with predefined number of values because 
        # the random values must be generated only one.
        return iter(qps_values), iter(latency_values)

    @classmethod
    def traffic_simulation(cls, 
                           qps_distribution, 
                           latency_distribution,
                           num_values):
        if qps_distribution == "random":
            qps_values = np.round(np.random.uniform(1, 100, num_values), 3)
            qps_values = qps_values.tolist()
        if latency_distribution == "random":
            latency_values = np.round(np.random.uniform(1, 5, num_values), 3)
            latency_values = latency_values.tolist()
        # We return a list with predefined number of values because 
        # the random values must be generated only one.            
        return iter(qps_values), iter(latency_values)

    @classmethod
    def execute_requests(cls, payloads, policy, max_workers=10):
        submitted_requests = []
        # Parallelize request submission.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for payload in payloads:
                if policy == "reactive":     
                    req = executor.submit(common.post_request_reactive, payload)
                    submitted_requests.append(req)
                else:
                    pass
        
        responses = []
        # Ensure requests completion.
        for task in as_completed(submitted_requests):
            responses.append(json.loads(task.result()))
        return responses

    @classmethod
    def generate_reactive_payload(cls,
                                  endpoint_id,
                                  timestamp,
                                  qps_iter,
                                  latency_iter):
        
        payload = copy.deepcopy(common.PAYLOAD_SCHEMA) # need deepcopy for randomization
        payload["endpoint_id"] = endpoint_id
        payload["total_requests"] = 0 # hardcoded number
        payload["policy"] = "reactive" # hardcoded policy        
        payload_data = copy.deepcopy(common.DATA_PAYLOAD_SCHEMA) # need deepcopy for randomization
        payload_data["timestamp"] = timestamp
        payload_data["latency"] = next(latency_iter)
        payload_data["qps"] = next(qps_iter)
        payload["data"] = [payload_data] # send a single data point
        return payload

    @classmethod
    def stress_test_reactive(cls,
                             num_endpoints=1,
                             warmup_requests_per_endpoint=100,
                             requests_per_endpoint=100,                         
                             submit_request_every_x_secs=60,
                             qps_distribution="random",
                             latency_distribution="random",
                             thread_pool_workers=10):
        logger.info(
            "Running reactive endpoint stress test with: {} endpoints, {} requests/endpoint, {} pool workers, \"{}\" qps distribution, \"{}\" latency distribution"
            .format(num_endpoints,
                    requests_per_endpoint, 
                    thread_pool_workers, 
                    qps_distribution, 
                    latency_distribution))        

        # Generate random endpoint ids. 
        endpoint_payloads = dict()
        for _ in range(num_endpoints):
            endpoint_id = random.randint(1, num_endpoints)                
            endpoint_payloads[endpoint_id] = []

        # For every endpoint submit a total number 
        # of `requests_per_endpoint` requests.
        for endpoint_id in endpoint_payloads.keys():
            current_timestamp = common.date_random(common.START_DATE)
            
            # Warmup Phase.
            warmup_qps_iter, warmup_latency_iter = \
                StressTest.warmup(warmup_requests_per_endpoint)
            for i in range(warmup_requests_per_endpoint):
                current_timestamp = common.date_increment_sec(
                    current_timestamp, secs=submit_request_every_x_secs) # linearly increasing timestamp every x-seconds
                timestamp = current_timestamp.strftime(common.CONFIG_DATETIME_FORMAT) # assign timestamp
                payload = StressTest.generate_reactive_payload(
                    endpoint_id, 
                    timestamp, 
                    warmup_qps_iter, 
                    warmup_latency_iter)
                endpoint_payloads[endpoint_id].append(payload)         

            # Simulate Traffic.
            dist_qps_iter, dist_latency_iter = \
                StressTest.traffic_simulation(
                    qps_distribution,
                    latency_distribution,
                    requests_per_endpoint)
            for i in range(requests_per_endpoint):
                current_timestamp = common.date_increment_sec(
                    current_timestamp, secs=submit_request_every_x_secs) # linearly increasing timestamp every x-seconds
                timestamp = current_timestamp.strftime(common.CONFIG_DATETIME_FORMAT) # assign timestamp                
                payload = StressTest.generate_reactive_payload(
                    endpoint_id,
                    timestamp,
                    dist_qps_iter,
                    dist_latency_iter)
                endpoint_payloads[endpoint_id].append(payload)
        
        st = time.time()
        endpoint_responses = dict()
        for endpoint_id, payloads in endpoint_payloads.items():
            responses = StressTest.execute_requests(payloads, policy="reactive")
            endpoint_responses[endpoint_id] = responses
        et = time.time()
        duration = et - st
        
        print("Num Endpoints: {} - NumRequestsPerEndpoint: {} - Time: {} secs."
            .format(num_endpoints, requests_per_endpoint * num_endpoints, duration))
        
        all_stress_tests = []
        for endpoint_id in endpoint_payloads:            
            all_stress_tests.append(
                StressTest.STRESS_TEST_STRUCTURE(
                    endpoint_id=endpoint_id,
                    payloads=payloads,
                    responses=endpoint_responses[endpoint_id]))
        return all_stress_tests

    classmethod
    def stress_test_predictive(cls):
        pass
