class BaseClientWorker(object):
    def __init__(self, client_index):
        self.client_index = client_index
        self.updated_information = 0

    def update(self, updated_information):
        self.updated_information = updated_information
        print(self.updated_information)

    def train(self):
        # complete your own algorithm operation here, as am example, we return the client_index
        return self.client_index
