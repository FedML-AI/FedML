import queue

# Sentinel for iterators stop
SENTINEL = None

def validate_client_id(func):
    def wrapper(self, client_id, *args, **kwargs):
        if self.send_queue_exits(client_id):
            return func(self, client_id, *args, **kwargs)
        else:
            raise "SomeName"

    return wrapper

class QueueManager(object):
    def __init__(self):
        self.send_messages_queues = {}
        self.received_messages_queue = queue.Queue()
    
    def send_queue_exits(self, client_id):
        return client_id in self.send_messages_queues

    def create_send_messages_queue(self, client_id):
        self.send_messages_queues[client_id] = queue.Queue()


    def add_to_received_messages_queue(self,message):
        self.received_messages_queue.put(message)

    @validate_client_id
    def add_to_send_messages_queue(self,client_id, message):
        self.send_messages_queues[client_id].put(message)

    def get_received_messages_iterator(self):
        return iter(self.received_messages_queue.get, SENTINEL)

    @validate_client_id
    def get_client_send_messages_iterator(self, client_id):
        return iter(self.send_messages_queues[client_id].get, SENTINEL)
    
    def stop_iterators(self):
        # End waiting for items in received messsages queue
        self.add_to_received_messages_queue(SENTINEL)

        # End waiting for items in send messsages queues
        for key in self.send_messages_queues.keys():
            self.add_to_send_messages_queue(key, SENTINEL)