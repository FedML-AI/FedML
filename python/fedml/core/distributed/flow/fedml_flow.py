import inspect
import logging
from time import sleep
from typing import Callable

from .fedml_executor import FedMLExecutor
from .fedml_flow_constants import (
    MSG_TYPE_CONNECTION_IS_READY,
    MSG_TYPE_NEIGHBOR_REPORT_NODE_STATUS,
    MSG_TYPE_NEIGHBOR_CHECK_NODE_STATUS,
    PARAMS_KEY_SENDER_ID,
    PARAMS_KEY_RECEIVER_ID,
    MSG_TYPE_FLOW_FINISH,
)
from ..communication.message import Message
from ..fedml_comm_manager import FedMLCommManager
from ...alg_frame.params import Params


class FedMLAlgorithmFlow(FedMLCommManager):
    ONCE = "FLOW_TAG_ONCE"
    FINISH = "FLOW_TAG_FINISH"

    def __init__(self, args, executor: FedMLExecutor):
        super().__init__(args, args.comm, args.rank, args.worker_num, args.backend)
        self.executor = executor
        self.executor_cls_name = self.executor.__class__.__name__
        logging.info("self.executor class name = {}".format(self.executor.__class__.__name__))

        self.flow_index = 0
        self.flow_sequence_original = []

        self.flow_sequence_current_map = dict()
        self.flow_sequence_next_map = dict()
        self.flow_sequence_executed = []

        # neighbor_node_online_status
        self.neighbor_node_online_map = dict()
        self.is_all_neighbor_connected = False

    def register_message_receive_handlers(self) -> None:
        self.register_message_receive_handler(MSG_TYPE_CONNECTION_IS_READY, self._handle_connection_ready)
        self.register_message_receive_handler(
            MSG_TYPE_NEIGHBOR_CHECK_NODE_STATUS, self._handle_neighbor_check_node_status,
        )
        self.register_message_receive_handler(
            MSG_TYPE_NEIGHBOR_REPORT_NODE_STATUS, self._handle_neighbor_report_node_status,
        )

        self.register_message_receive_handler(
            MSG_TYPE_FLOW_FINISH, self._handle_flow_finish,
        )

        for flow_idx in range(len(self.flow_sequence_original) - 1):
            (flow_name, executor_task, executor_task_cls_name, flow_tag,) = self.flow_sequence_original[flow_idx]
            (
                flow_name_next,
                executor_task_next,
                executor_task_cls_name_next,
                flow_tag_next,
            ) = self.flow_sequence_next_map[flow_name]
            if executor_task_cls_name_next == self.executor_cls_name:
                logging.info("self.register_message_receive_handler. msg_type = {}".format(flow_name))
                self.register_message_receive_handler(flow_name, self._handle_message_received)

    def add_flow(self, flow_name, executor_task: Callable, flow_tag=ONCE):

        logging.info("flow_name = {}, executor_task = {}".format(flow_name, executor_task))
        executor_task_cls_name = self._get_class_that_defined_method(executor_task)
        logging.info("executor_task class name = {}".format(executor_task_cls_name))
        self.flow_sequence_original.append((flow_name + str(self.flow_index), executor_task, executor_task_cls_name, flow_tag))
        self.flow_index += 1

    def run(self):
        super().run()

    def build(self):
        logging.info("self.flow_sequence = {}".format(self.flow_sequence_original))
        (flow_name, executor_task, executor_task_cls_name, flow_tag,) = self.flow_sequence_original[
            len(self.flow_sequence_original) - 1
        ]
        self.flow_sequence_original[len(self.flow_sequence_original) - 1] = (
            flow_name,
            executor_task,
            executor_task_cls_name,
            FedMLAlgorithmFlow.FINISH,
        )

        for flow_idx in range(len(self.flow_sequence_original)):
            (flow_name, executor_task, executor_task_cls_name, flow_tag,) = self.flow_sequence_original[flow_idx]
            self.flow_sequence_current_map[flow_name] = (
                flow_name,
                executor_task,
                executor_task_cls_name,
                flow_tag,
            )

            if flow_idx == len(self.flow_sequence_original) - 1:
                self.flow_sequence_next_map[flow_name] = (None, None, None, None)
                break

            (
                flow_name_next,
                executor_task_next,
                executor_task_cls_name_next,
                flow_tag_next,
            ) = self.flow_sequence_original[flow_idx + 1]
            self.flow_sequence_next_map[flow_name] = (
                flow_name_next,
                executor_task_next,
                executor_task_cls_name_next,
                flow_tag_next,
            )
        logging.info("self.flow_sequence_next_map = {}".format(self.flow_sequence_next_map))

    def _on_ready_to_run_flow(self):
        logging.info("#######_on_ready_to_run_flow#######")
        (
            flow_name_current,
            executor_task_current,
            executor_task_cls_name_current,
            flow_tag_current,
        ) = self.flow_sequence_original[0]
        if self.executor_cls_name == executor_task_cls_name_current:
            self._execute_flow(
                None, flow_name_current, executor_task_current, executor_task_cls_name_current, flow_tag_current
            )

    def _handle_message_received(self, msg_params):
        flow_name = msg_params.get_type()

        flow_params = Params()
        for key in msg_params.get_params():
            flow_params.add(key, msg_params.get_params()[key])
        logging.info("_handle_message_received. flow_name = {}".format(flow_name))
        (flow_name, executor_task, executor_task_cls_name, flow_tag) = self.flow_sequence_current_map[flow_name]
        (flow_name_next, executor_task_next, executor_task_cls_name_next, flow_tag_next) = self.__direct_to_next_flow(
            flow_name, flow_tag
        )

        self._execute_flow(flow_params, flow_name_next, executor_task_next, executor_task_cls_name_next, flow_tag_next)

    def _execute_flow(self, flow_params, flow_name, executor_task, executor_task_cls_name, flow_tag):
        logging.info(
            "\n\n###########_execute_flow (START). flow_name = {}, executor_task name = {}() #######".format(
                flow_name, executor_task.__name__
            )
        )
        self.executor.set_params(flow_params)
        if self.executor_cls_name != executor_task_cls_name:
            raise Exception(
                "The current executor cannot execute a task in a different executor. executed flow = {}".format(
                    self.flow_sequence_executed
                )
            )
        params = executor_task(self.executor)
        logging.info(
            "\n###########_execute_flow (END). flow_name = {}, executor_task name = {}() #######\n\n".format(
                flow_name, executor_task.__name__
            )
        )
        self.flow_sequence_executed.append(flow_name)
        (flow_name_next, executor_task_next, executor_task_cls_name_next, flow_tag_next,) = self.__direct_to_next_flow(
            flow_name, flow_tag
        )
        if flow_name_next is None:
            logging.info("FINISHED")
            # broadcast FINISH message
            self.__shutdown()
            return
        if params is None:
            logging.info("terminate propagation")
            return
        params.add(PARAMS_KEY_SENDER_ID, self.executor.get_id())
        if executor_task_cls_name_next == self.executor_cls_name:
            params.add(PARAMS_KEY_RECEIVER_ID, [self.executor.get_id()])
            # call locally
            logging.info("flow_name = {}, receive_id = {}".format(flow_name, [self.executor.get_id()]))
            self._pass_message_locally(flow_name, params)
        else:
            params.add(PARAMS_KEY_RECEIVER_ID, self.executor.get_neighbor_id_list())
            logging.info("flow_name = {}, receive_id = {}".format(flow_name, self.executor.get_neighbor_id_list()))
            self._send_msg(flow_name, params)

    def __direct_to_next_flow(self, flow_name, flow_tag):
        (
            flow_name_next,
            executor_task_next,
            executor_task_cls_name_next,
            flow_tag_next,
        ) = self.flow_sequence_next_map[flow_name]
        return (
            flow_name_next,
            executor_task_next,
            executor_task_cls_name_next,
            flow_tag_next,
        )

    def _send_msg(self, flow_name, params: Params):
        sender_id = params.get(PARAMS_KEY_SENDER_ID)
        receiver_id = params.get(PARAMS_KEY_RECEIVER_ID)
        logging.info("sender_id = {}, receiver_id = {}".format(sender_id, receiver_id))
        for rid in receiver_id:
            message = Message(flow_name, sender_id, rid)
            logging.info("params.keys() = {}".format(params.keys()))
            logging.info("params.values() = {}".format(params.values()))
            for key in params.keys():
                if key == Message.MSG_ARG_KEY_TYPE:
                    continue
                message.add_params(key, params.get(key))
            self.send_message(message)

    def _handle_flow_finish(self, msg_params):
        self.__shutdown()

    def __shutdown(self):
        for rid in self.executor.get_neighbor_id_list():
            message = Message(MSG_TYPE_FLOW_FINISH, self.executor.get_id(), rid)
            self.send_message(message)
        sleep(1)
        self.finish()

    def _pass_message_locally(self, flow_name, params: Params):
        sender_id = params.get(PARAMS_KEY_SENDER_ID)
        receiver_id = params.get(PARAMS_KEY_RECEIVER_ID)
        logging.info("sender_id = {}, receiver_id = {}".format(sender_id, receiver_id))
        for rid in receiver_id:
            message = Message(flow_name, sender_id, rid,)
            logging.info("params.keys() = {}".format(params.keys()))
            for key in params.keys():
                if key == Message.MSG_ARG_KEY_TYPE:
                    continue
                value = params.get(key)
                message.add_params(key, value)
            self._handle_message_received(message)

    def _handle_connection_ready(self, msg_params):
        if self.is_all_neighbor_connected:
            return
        logging.info("_handle_connection_ready")
        for receiver_id in self.executor.get_neighbor_id_list():
            self._send_message_to_check_neighbor_node_status(receiver_id)
            self._send_message_to_report_node_status(receiver_id)

    def _handle_neighbor_report_node_status(self, msg_params):
        sender_id = msg_params.get_sender_id()
        logging.info(
            "_handle_neighbor_report_node_status. node_id = {}, neighbor_id = {} is online".format(
                self.executor.get_id(), sender_id
            )
        )
        self.neighbor_node_online_map[str(sender_id)] = True

        all_neighbor_nodes_is_online = True
        for neighbor_id in self.executor.get_neighbor_id_list():
            if not self.neighbor_node_online_map.get(str(neighbor_id), False):
                all_neighbor_nodes_is_online = False
                break
        if all_neighbor_nodes_is_online:
            self.is_all_neighbor_connected = True
        if self.is_all_neighbor_connected:
            self._on_ready_to_run_flow()

    def _handle_neighbor_check_node_status(self, msg_params):
        sender_id = msg_params.get_sender_id()
        self._send_message_to_report_node_status(sender_id)

    def _send_message_to_check_neighbor_node_status(self, receiver_id):
        message = Message(MSG_TYPE_NEIGHBOR_CHECK_NODE_STATUS, self.executor.get_id(), receiver_id)
        logging.info(
            "_send_message_to_check_neighbor_node_status. node_id = {}, neighbor_id = {} is online".format(
                self.executor.get_id(), receiver_id
            )
        )
        self.send_message(message)

    def _send_message_to_report_node_status(self, receiver_id):
        message = Message(MSG_TYPE_NEIGHBOR_REPORT_NODE_STATUS, self.executor.get_id(), receiver_id)
        self.send_message(message)

    def _get_class_that_defined_method(self, meth):
        if inspect.ismethod(meth):
            for cls in inspect.getmro(meth.__self__.__class__):
                if cls.__dict__.get(meth.__name__) is meth:
                    return cls.__name__
            meth = meth.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(meth):
            class_name = meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0]
            try:
                cls = getattr(inspect.getmodule(meth), class_name)
            except AttributeError:
                cls = meth.__globals__.get(class_name)
            if isinstance(cls, type):
                return cls.__name__
        return None  # not required since None would have been implicitly returned anyway
