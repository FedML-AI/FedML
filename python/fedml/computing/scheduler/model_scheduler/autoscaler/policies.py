from pydantic import BaseModel, NonNegativeInt, NonNegativeFloat, validator


class AutoscalingPolicy(BaseModel):
    """
    Below are some default values for every endpoint.

    The following parameters refer to:
    - current_replicas: the number of currently running replicas of the endpoint
    - min_replicas: the minimum number of replicas of the endpoint in the instance group
    - max_replicas: the maximum number of replicas of the endpoint in the instance group
    - release_replica_after_idle_secs: when to release a single idle replica
    - scaledown_delay_secs: how many seconds to wait before performing a scale down operation
    - scaleup_cost_secs: how many seconds it takes/costs to perform a scale up operation
    - previous_triggering_value: the last value that triggered a scaling operation

    The `replica_idle_grace_secs` parameter is used as
    the monitoring interval after which a running replica
    of an idle endpoint should be released.
    """
    current_replicas: NonNegativeInt
    min_replicas: NonNegativeInt
    max_replicas: NonNegativeInt
    previous_triggering_value: float = None
    release_replica_after_idle_secs: NonNegativeInt = None
    scaledown_delay_secs: NonNegativeInt = 60  # default is 1 minute
    scaleup_cost_secs: NonNegativeInt = 300  # default is 5 minutes


class EWMPolicy(AutoscalingPolicy):
    """
    Configuration parameters for the reactive autoscaling policy.
    EWM stands for Exponential Weighted Calculations, since we use
    the pandas.DataFrame.ewm() functionality.

    For panda's EWM using alpha = 0.1, we indicate that the most recent
    values are weighted more. The reason is that the exponential weighted
    mean formula in pandas is computed as:
        Yt = X_t + (1-a) * X_{t-1} + (1-a)^2 X_{t-2} / (1 + (1-a) + (1-a)^2)

    The following parameters refer to:
    - ewm_mins: the length of the interval we consider for reactive decision
    - ewm_alpha: the decay factor for the exponential weighted interval
    - ewm_latest: the latest recorded value of the metric's exponential weighted mean
    - ub_threshold: the upper bound scaling factor threshold for reactive decision
    - lb_threshold: the lower bound scaling factor threshold for reactive decision

    Example:

        Let's say that we consider 15 minutes as the length of our interval and a
        decay factor alpha with a value of 0.5:
            Original Sequence: [0.1, 0.2, 0.4, 3, 5, 10]
            EWM Sequence: [0.1, [0.166, 0.3, 1.74, 3.422, 6.763]

        If we assume that our previous scaling operation was triggered at value Y,
        then the conditions we use to decide whether to scale up or down are:
            Latency:
                ScaleUP: X > ((1 + ub_threshold) * Y)
                ScaleDown: X < (lb_threshold * Y)
            QPS:
                ScaleUP: X < (lb_threshold * Y)
                ScaleDown: X < ((1 + ub_threshold) * Y)

        In other words, QPS is the inverse of Latency and vice versa.
    """
    metric: str  # possible values: ["ewm_latency", "ewm_qps"]
    ewm_mins: NonNegativeInt  # recommended value: 15 minutes
    ewm_alpha: NonNegativeFloat  # recommended value: 0.1
    ewm_latest: NonNegativeFloat = None  # will be filled by the algorithm
    ub_threshold: NonNegativeFloat  # recommended value: 0.5
    lb_threshold: NonNegativeFloat  # recommended value: 0.5

    @validator("metric")
    def metric_match(cls, v) -> str:
        if v not in ["ewm_latency", "ewm_qps"]:
            raise ValueError("Wrong metric name.")
        return v


class ConcurrentQueryPolicy(AutoscalingPolicy):
    """
    This policy captures the number of queries we want to support
    per replica over the defined window length in seconds.
    """
    queries_per_replica: NonNegativeInt  # recommended is at least 1 query
    window_size_secs: NonNegativeInt  # recommended is at least 60seconds


class MeetTrafficDemandPolicy(AutoscalingPolicy):
    """
    This policy captures the number of queries we want to support
    per replica over the defined window length in seconds.
    """
    window_size_secs: NonNegativeInt


class PredictivePolicy(AutoscalingPolicy):
    # TODO(fedml-dimitris): TO BE COMPLETED!
    pass
