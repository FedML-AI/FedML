from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from fedml.core.security.fedml_defender import FedMLDefender
from fedml.core.security.fedml_attacker import FedMLAttacker
from fedml.core.dp.fed_privacy_mechanism import FedMLDifferentialPrivacy
from fedml.ml.aggregator.agg_operator import FedMLAggOperator


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
        self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> List[Tuple[float, Dict]]:
        if FedMLAttacker.get_instance().is_model_attack():
            raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
                local_w=raw_client_model_or_grad_list,
                global_w=self.get_model_params(),
                refs=None,
            )
        if FedMLDefender.get_instance().is_defense_enabled():
            raw_client_model_or_grad_list = (
                FedMLDefender.get_instance().defend_before_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
            )

        return raw_client_model_or_grad_list

    def aggregate(
        self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> Dict:
        if FedMLDefender.get_instance().is_defense_enabled():
            return FedMLDefender.get_instance().defend_on_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                base_aggregation_func=FedMLAggOperator.agg,
                extra_auxiliary_info=self.get_model_params(),
            )
        return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: Dict) -> Dict:
        if FedMLDifferentialPrivacy.get_instance().is_dp_enabled():
            aggregated_model_or_grad = FedMLDifferentialPrivacy.get_instance().add_cdp_noise(
                aggregated_model_or_grad
            )
        return aggregated_model_or_grad

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args
    ) -> bool:
        pass
