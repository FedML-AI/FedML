from ..common.singleton import Singleton


class MLOpsStatus(Singleton):
    _status_instance = None

    def __init__(self):
        self.messenger = None
        self.run_id = None
        self.edge_id = None
        self.client_agent_status = dict()
        self.server_agent_status = dict()
        self.client_status = dict()
        self.server_status = dict()

    @staticmethod
    def get_instance():
        if MLOpsStatus._status_instance is None:
            MLOpsStatus._status_instance = MLOpsStatus()

        return MLOpsStatus._status_instance

    def set_client_agent_status(self, edge_id, status):
        self.client_agent_status[edge_id] = status

    def set_server_agent_status(self, edge_id, status):
        self.server_agent_status[edge_id] = status

    def set_client_status(self, edge_id, status):
        self.client_status[edge_id] = status

    def set_server_status(self, edge_id, status):
        self.server_status[edge_id] = status

    def get_client_agent_status(self, edge_id):
        return self.client_agent_status.get(edge_id, None)

    def get_server_agent_status(self, edge_id):
        return self.server_agent_status.get(edge_id, None)

    def get_client_status(self, edge_id):
        return self.client_status.get(edge_id, None)

    def get_server_status(self, edge_id):
        return self.server_status.get(edge_id, None)
