from typing import Any, List, Optional

from dataclasses import dataclass, field
import warnings

from fedml.train.llm.configurations import ExperimentArguments
from fedml.train.llm.integrations import is_fedml_available
import torch.cuda

if is_fedml_available():
    from fedml.arguments import Arguments


@dataclass
class UnitedLLMExperimentArguments(ExperimentArguments):
    # distributed
    unitedllm_rank: int = field(default=0, metadata={"help": "Distributed rank for UnitedLLM."})
    role: str = field(
        default="client",
        metadata={"help": "Number of communication rounds.", "choices": ["client", "server"]}
    )
    use_customized_hierarchical: bool = field(
        default=True,
        metadata={"help": "Whether to use customized hierarchical distributions."}
    )
    # aggregation
    client_num_in_total: int = field(default=1, metadata={"help": "Number of clients."})
    client_num_per_round: int = field(default=1, metadata={"help": "Number of clients participate in aggregation."})
    comm_round: int = field(default=5, metadata={"help": "Number of communication rounds."})
    # training
    local_num_train_epochs: float = field(
        default=-1.0,
        metadata={
            "help": "Total number of training epochs to perform for each communication round. If set to a positive"
                    " value, overrides `num_train_epochs` with `num_train_epochs = local_num_train_epochs *"
                    " comm_round`.",
        }
    )
    local_max_steps: int = field(
        default=-1,
        metadata={
            "help": "Total number of training steps to perform for each communication round. If set to a positive,"
                    " override `max_steps`, `local_num_train_epochs` and `num_train_epochs`; set `max_steps` to"
                    " `max_steps = local_max_steps * comm_round`."
        },
    )
    # evaluation
    frequency_of_the_test: int = field(
        default=1,
        metadata={"help": "Number of communication rounds between two evaluations."}
    )
    test_on_clients: str = field(
        default="no",
        metadata={
            "help": "Timing for evaluation on clients.",
            "choices": ["before_aggregation", "after_aggregation", "no", "both"],
        }
    )
    test_on_client_ranks: List[int] = field(
        default_factory=list,
        metadata={"help": "The client ranks to run evaluations.", "nargs": "+"}
    )
    # checkpoint saving
    save_frequency: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of updates steps before two checkpoint saves. Set to 0 to disable saving."
                    " Set to a negative number or None to save after every test (i.e. same as"
                    " `frequency_of_the_test`).",
        }
    )
    # optimization
    federated_optimizer: str = field(
        default="FedAvg",
        metadata={"help": "Aggregation optimizer for the aggregator.", "choices": ["FedAvg"]}
    )
    # optional
    _fedml_args: Optional["Arguments"] = field(
        default=None,
        init=False,
        metadata={
            "help": "Reference to the `fedml.arguments.Arguments` object. This should be added by calling "
                    "`add_and_verify_fedml_args`"
        }
    )

    def __post_init__(self) -> None:
        if not is_fedml_available():
            raise RuntimeError("UnitedLLMExperimentArguments requires fedml to be installed. Run `pip install fedml`.")

        if self.save_frequency is None or self.save_frequency < 0:
            self.save_frequency = self.frequency_of_the_test

        if torch.cuda.device_count() == 0:
            warnings.warn(f"{self.role} rank {self.unitedllm_rank} does not have GPU. Fallback to CPU mode.")
            self.deepspeed = None

        if self.role == "client" and not self.should_evaluate:
            # disable when not testing on this client
            self.report_to = ["none"]
            self.disable_tqdm = True

        if self.comm_round <= 0:
            raise ValueError(f"comm_round must be at least 1 but received {self.comm_round}")

        if self.local_max_steps <= 0 and self.local_num_train_epochs <= 0:
            raise ValueError(
                f"At least 1 of `local_max_steps` and `local_num_train_epochs` should be positive, "
                f"but received {self.local_max_steps} and {self.local_num_train_epochs}."
            )

        # update `num_train_epochs` and `max_steps`
        self.num_train_epochs = max(self.local_num_train_epochs * self.comm_round, -1.0)
        self.max_steps = max(self.local_max_steps * self.comm_round, -1)

        super().__post_init__()

    def add_and_verify_args(self, *args: Any) -> None:
        for args_obj in args:
            if is_fedml_available() and isinstance(args_obj, Arguments):
                self.add_and_verify_fedml_args(args_obj)
            else:
                super().add_and_verify_args(args_obj)

    @property
    def should_evaluate(self) -> bool:
        return (
                self.role != "client" or
                (self.unitedllm_rank in self.test_on_client_ranks and self.test_on_clients != "no")
        )

    def add_and_verify_fedml_args(self, fedml_args: "Arguments") -> None:
        if fedml_args.rank != self.unitedllm_rank:
            raise ValueError(
                f"Conflicting rank detected. fedml_args.rank = {fedml_args.rank} while"
                f" unitedllm_rank = {self.unitedllm_rank}"
            )

        if fedml_args.role != self.role:
            raise ValueError(
                f"Conflicting role detected. fedml_args.role = {fedml_args.role} while"
                f" role = {self.role}"
            )

        # synchronize configs
        setattr(fedml_args, "frequency_of_the_test", self.frequency_of_the_test)
        setattr(fedml_args, "save_frequency", self.save_frequency)
        setattr(fedml_args, "use_customized_hierarchical", self.use_customized_hierarchical)
        setattr(fedml_args, "num_train_epochs", self.num_train_epochs)
        setattr(fedml_args, "max_steps", self.max_steps)

        # update cross-silo hierarchical related settings
        if self.use_customized_hierarchical:
            setattr(fedml_args, "proc_rank_in_silo", self.process_index)
            setattr(fedml_args, "rank_in_node", self.local_process_index)
            setattr(fedml_args, "process_id", self.process_index)

        self._fedml_args = fedml_args
