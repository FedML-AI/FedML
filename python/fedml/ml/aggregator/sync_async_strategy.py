import sys
from abc import ABC, abstractmethod


class BaseAggStrategy(ABC):
    """
    Abstract base class for aggregation strategy
    i.e., async and sync aggregation and their variants.

    Terminology:
        buffer size - aggregation server needs to maintain a buffer for client updates, once it arrives at the "buffer size",
        the aggregation operation will be executed. In synchronous distributed optimization, buffer size is equal to the number
        of clients, while in asynchronous distributed optimization, it's much smaller than the number of clients.

        staleness - staleness is a concept in asynchronous distributed optimization: the optimization parameter (weights) is
        updated using stale gradients, which are gradients calculated based on out-of-date aggregated weights; staleness represents
        how many aggregation steps the outdate has.

        staleness scaling factor - To control the effect of staleness in a client’s contribution to the server aggregation, we
        need to down-weight stale updates using a specific function: "staleness scaling factor" = func(staleness).

        max_acceptable_staleness - Once the staleness of client update is equal or larger than the max_acceptable_staleness,
        we will drop the client update.
    """

    def __init__(self, config):
        self.config = config

        self.is_async = False
        if (hasattr(config, "is_async") and isinstance(config.is_async, bool) and config.is_async):
            self.is_async = True

        if self.is_async:
            # set the config.buffer_size
            if hasattr(config, "buffer_size") and isinstance(config.buffer_size, int):
                if (config.buffer_size >= config.client_num_per_round or config.buffer_size <= 0):
                    raise AttributeError(
                        f"config.buffer_size ({config.buffer_size}) is abnormal. In async optimization, buffer_size should be < config.client_num_per_round")
                else:
                    self.buffer_size = config.buffer_size
            else:
                raise AttributeError(f"config.buffer_size ({config.buffer_size}) is not set.")

            # set the config.max_acceptable_staleness
            if hasattr(config, "max_acceptable_staleness") and isinstance(config.max_acceptable_staleness, int):
                if config.max_acceptable_staleness < 0:
                    raise AttributeError(
                        f"config.max_acceptable_staleness ({config.max_acceptable_staleness}) is abnormal.")
                else:
                    self.max_acceptable_staleness = config.max_acceptable_staleness
            else:
                # when config.max_acceptable_staleness is not set,
                # we set it to infinite int value to allow accepting all client updates.
                config.max_acceptable_staleness = sys.maxsize
        else:
            self.buffer_size = config.client_num_per_round

        # count how many updates have been inserted into the buffer, used by the server
        # we use set() to avoid duplidate updates
        self.client_update_index_in_buffer = set()

    def is_asynchronous(self):
        return self.is_async

    def get_buffer_size(self):
        return self.buffer_size

    @abstractmethod
    def whether_to_accept(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        pass

    def add_client_update_index_to_buffer(self, client_update_index):
        if client_update_index not in self.client_update_index_in_buffer:
            self.client_update_index_in_buffer.add(client_update_index)

    def whether_to_aggregate(self):
        if len(self.client_update_index_in_buffer) < self.buffer_size:
            return False
        return True

    @abstractmethod
    def get_weight_scaling_ratio(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        pass
    
    def get_client_update_index_in_buffer(self):
        return self.client_update_index_in_buffer
    
    def reset_buffer(self):
        self.client_update_index_in_buffer.clear()


class SyncAggStrategy(BaseAggStrategy):
    def __init__(self, config):
        super().__init__(config)

    def whether_to_accept(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        return True

    def get_weight_scaling_ratio(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        return 1


class FedBuffAsyncAggStrategy(BaseAggStrategy):
    def __init__(self, config):
        super().__init__(config)

    def whether_to_accept(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        staleness = current_global_step_on_server - current_global_step_on_client
        return staleness <= self.max_acceptable_staleness

    def get_weight_scaling_ratio(
        self, current_global_step_on_server, current_global_step_on_client
    ):
        staleness = current_global_step_on_server - current_global_step_on_client
        return 1 / (1 + staleness) ** 0.5


def create_agg_strategy(config):
    if config.federated_optimizer == "FedBuff":
        config.is_async = True
        return FedBuffAsyncAggStrategy(config)
    else:
        config.is_async = False
        return SyncAggStrategy(config)
