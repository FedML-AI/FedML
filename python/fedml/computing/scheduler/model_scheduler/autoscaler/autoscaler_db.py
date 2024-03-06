import conf
import json
import sqlite3

from collections import defaultdict
from utils.logger import logger
from utils.date import convert_to_datetime
from utils.singleton import Singleton


class AutoscalerKVCache(metaclass=Singleton):

    def __init__(self, values_per_key) -> None:
        self.values_per_key = values_per_key
        self.__state = defaultdict(list)
    
    @staticmethod
    def get_instance(values_per_key):
        instance = AutoscalerKVCache(values_per_key)    
        return instance

    def bulk_insert(self, records):
        # Insert all records at once.
        for rec in records:
            for k, v in rec.items():
                self.__state[k].extend(v)
        # Ensure the size of the list values
        # conforms to the expected cache size.
        self.maintain_cache_size([k for k in self.__state.keys()])

    def insert(self, k, v):
        """
        The input value v is a list with the following format:
            [
                {'timestamp': '2024-01-26T13:23:13z', 'latency': 1.615, 'qps': 43.458},
                {},
                ...
            ]
        """
        assert isinstance(v, list)
        self.__state[k].extend(v)
        self.maintain_cache_size([k])

    def get_by_key(self, k, sorted=True):
        if sorted:
            return self.sort_by_timestamp(k)
        else:
            return self.__state[k]
    
    def get_all(self, sorted=True):
        all_data = []
        for k in self.__state:
            if sorted:
                v = self.sort_by_timestamp(k)
            all_data.append(v)
        return all_data

    def maintain_cache_size(self, keys):
        assert isinstance(keys, list)
        for k in keys:
            sorted_values = self.sort_by_timestamp(k)
            if len(sorted_values) > self.values_per_key:
                self.__state[k] = self.__state[k][-self.values_per_key:]

    def sort_by_timestamp(self, k):
        sorted_by_timestamp = \
            sorted(self.__state[k], 
                   key=lambda d: convert_to_datetime(
                       d["timestamp"], 
                       conf.CONFIG_DATETIME_FORMAT))
        return sorted_by_timestamp

    def delete(self, k):
        self.__state.pop(k, None)

    def erase(self):
        self.__state = defaultdict(list)


class AutoscalerDiskDB(metaclass=Singleton):
    
    """
    We are not intializing the Database Class as a fedml.core.common.singleton because 
    that Singleton approach does not allow to pass arguments during initialization. 
    In particular, the error that is raised with the previous approach is:
        `TypeError: object.__new__() takes exactly one argument (the type to instantiate)`
    """

    def __init__(self, dbpath):
        self.dbpath = dbpath
        # We pass the PARSE_DECLTYPES to allow storage 
        # and retrieval of datatime.datetime data types.
        self.sqlite3_conn = \
            lambda: sqlite3.connect(self.dbpath, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            with self.sqlite3_conn() as conn:
                self.create_query_reply_tbl(conn)
                self.create_query_data_tbl(conn)
            logger.info("AutoscalerDB path: {}".format(self.dbpath))
        except sqlite3.Error as e:
            logger.error("AutoscalerDB initialization error: {}".format(e))
    
    @staticmethod
    def get_instance(dbpath):
        instance = AutoscalerDiskDB(dbpath)
        return instance

    def create_query_reply_tbl(self, conn):
        # Since an query might have multiple data values we do not store these 
        # values inside the current table. Rather, we create a separate table 
        # where the association key is the automatically generated query_id!
        # Moreover to simplify analysis in this table we store both the body 
        # of the request along with the reply of the autoscaler on every 
        # incoming request.
        create_tbl_stmt = """
            CREATE TABLE IF NOT EXISTS QueryReply (
                qid INTEGER PRIMARY KEY AUTOINCREMENT,
                query_endpoint_id INTEGER NOT NULL,                
                query_total_requests INTEGER,
                query_policy_type TEXT NOT NULL,
                reply_allocate_instance INTEGER,
                reply_free_instance INTEGER,
                reply_trigger_predictive INTEGER);
        """
        curs = conn.cursor()
        curs.execute(create_tbl_stmt)

    def insert_query_reply_tbl(self, data, conn):
        # Sicne qid is auto-incrementing we do not have to pass 
        # a specific value, NULL will generate one automatically.
        insert_tbl_stmt = """
            INSERT INTO QueryReply VALUES
                (NULL, :endpoint_id, :total_requests, :policy_type, :allocate_instance, :free_instance, :trigger_predictive);
        """
        curs = conn.cursor()
        curs.execute(insert_tbl_stmt, data)
        qid = curs.lastrowid
        return qid

    def create_query_data_tbl(self, conn):
        # The query id references the query id of the "parent" QueryReply table.
        create_tbl_stmt = """
            CREATE TABLE IF NOT EXISTS QueryData ( \
                qid INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                latency REAL,
                qps REAL,                
            FOREIGN KEY (qid) REFERENCES QueryReply(qid));
        """
        # We do not create a UNIQUE index because some timestamps may 
        # overlap for different queries and the UNIQUE index expects 
        # unique values over the columns's (i.e., timestamp) values. 
        idx_stmt = """
            CREATE INDEX IF NOT EXISTS idx_timestamp ON QueryData (timestamp);
        """
        curs = conn.cursor()
        curs.execute(create_tbl_stmt)
        curs.execute(idx_stmt)

    def insert_query_data_tbl(self, data, conn):
        insert_tbl_stmt = """
            INSERT INTO QueryData 
                (qid, timestamp, latency, qps) VALUES 
                (:qid, :timestamp, :latency, :qps); 
        """
        
        curs = conn.cursor()
        # The data parameter might be a sequence of dictionaries:
        # e.g., [{}, {}, {}], hence the execute_many() command.
        curs.executemany(insert_tbl_stmt, data)

    def populate_db(self,
                    endpoint_id,
                    total_requests,
                    policy_type, 
                    allocate_instance,
                    free_instance,
                    trigger_predictive,
                    data):
        with self.sqlite3_conn() as conn:
            qid = self.insert_query_reply_tbl({
                "endpoint_id": endpoint_id,
                "total_requests": total_requests,
                "policy_type": policy_type,
                "allocate_instance": allocate_instance,
                "free_instance": free_instance,
                "trigger_predictive": trigger_predictive
            }, conn)
            data_augmented = [dict(item, **{"qid":qid}) for item in data]
            self.insert_query_data_tbl(data_augmented, conn)

    def select_num_records_per_table(self):
        with sqlite3.connect(self.dbpath) as conn:
            cur = conn.cursor()
            query_reply_num = \
                cur.execute("SELECT count(*) FROM QueryReply;").fetchone()
            query_data_num = \
                cur.execute("SELECT count(*) FROM QueryData;").fetchone()
            return {"QueryReplyCount": query_reply_num, "QueryDataCount": query_data_num}
    
    def select_query_data(self, records_per_query):
        """
        1. Get all query ids that are associated to each combo
            (model_id, endpoint_id, edge_id)
        2. Retrieve the most recent `records_per_query` data per combo
        3. Cast each returned datum to its json counterpart
        4. Return the respective data for each combo
        """

        with self.sqlite3_conn() as conn:
            cur = conn.cursor()            
            query_ids_per_endpoint = """ 
                SELECT DISTINCT
                    query_endpoint_id,
                    GROUP_CONCAT(qid, ',')            
                FROM QueryReply
                GROUP BY query_endpoint_id;
            """
            query_ids_data = """
                SELECT 
                    json_object(
                        'timestamp', timestamp,
                        'qps', qps,
                        'latency', latency)
                FROM QueryData as qd
                WHERE qd.qid IN ({})
                ORDER BY timestamp DESC
                LIMIT {}; """
            
            all_query_data = []
            query_ids_res = cur.execute(query_ids_per_endpoint).fetchall()
            for res_1 in query_ids_res:             
                res_1_json = {
                    "endpoint_id": res_1[0],
                }
                qids = res_1[1]
                query_data_res = cur.execute(query_ids_data.format(qids, records_per_query)) 
                for res_2 in query_data_res:
                    res_2_json = json.loads(res_2[0])
                    res_2_json.update(res_1_json) # combine both json results
                    all_query_data.append(res_2_json)
            
            return all_query_data        
        