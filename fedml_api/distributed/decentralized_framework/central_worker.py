import logging


class BaseCentralWorker(object):
    def __init__(self, client_num, args):
        self.client_num = client_num
        self.args = args

        self.client_local_result_list = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def add_client_local_result(self, index, client_local_result):
        logging.info("add_client_local_result. index = %d" % index)
        self.client_local_result_list[index] = client_local_result
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        global_result = 0
        for k in self.client_local_result_list.keys():
            global_result += self.client_local_result_list[k]
        return global_result
