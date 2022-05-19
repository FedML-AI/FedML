from ...core.distributed.client.client_manager import ClientManager


class CommunicationManager(ClientManager):
    def __init__(self, args, comm, rank, size, backend):
        super().__init__(args, comm, rank, size, backend)

    def register_message_receive_handlers(self):
        pass

    def run(self):
        super().run()
