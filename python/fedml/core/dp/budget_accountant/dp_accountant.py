import enum
from typing import Optional
import numpy as np
import warnings
import math
import scipy.optimize as sp_opt
from fedml.core.dp.budget_accountant import rdp_privacy_accountant
from fedml.core.dp.budget_accountant import dp_event

"""Keeping track of the differential privacy guarantee."""


class NeighboringRelation(enum.Enum):
    ADD_OR_REMOVE_ONE = 1
    REPLACE_ONE = 2

    # A record is replaced with a special record, such as the "zero record". See
    # https://arxiv.org/pdf/2103.00039.pdf, Definition 1.1.
    REPLACE_SPECIAL = 3


def calibrate_steps(target_epsilon: float, target_delta: float, noise_multipliers: float, batch_sizes: int,
                    num_examples: int, initial_estimate: int = 4, initial_min_steps: int = 1, tol: float = 0.1):
    def get_epsilon(steps):
        return compute_epsilon(
            noise_multipliers,
            batch_sizes,
            steps,
            num_examples,
            target_delta,
        )

    if get_epsilon(initial_min_steps) > target_epsilon:
        raise ValueError('Epsilon at initial_min_steps is too large. '
                         'Try increasing `target_epsilon`.')

    max_steps = initial_estimate
    min_steps = initial_min_steps

    while get_epsilon(max_steps) < target_epsilon:
        min_steps, max_steps = max_steps, 2 * max_steps

    error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
    opt_result = sp_opt.minimize_scalar(
        error_epsilon,
        bounds=(min_steps, max_steps),
        method='bounded',
        options={'xatol': tol},
    )
    assert opt_result.success

    return math.ceil(opt_result.x)


def calibrate_noise_multiplier(
        target_epsilon: float,
        batch_sizes: int,
        num_steps: int,
        num_examples: int,
        target_delta: float = 1e-5,
        tol: float = 0.01,
) -> float:
    """Computes the noise multiplier to achieve `target_epsilon`.

    Args:
      target_epsilon: The desired final epsilon.
      batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
        noise multiplier changes across steps. 't' indicates step where batch_size
        is set to 'bs'.
      num_steps: Total number of iterations.
      num_examples: Number of training examples.
      target_delta: Desired delta for the returned epsilon.
      tol: tolerance of the optimizer for the calibration.

    Returns:
      noise_multiplier: Noise multiplier.
    """

    def get_epsilon(noise_multiplier):
        return compute_epsilon(
            noise_multiplier,
            batch_sizes,
            num_steps,
            num_examples,
            target_delta)

    max_noise = 1.0
    min_noise = 0.0
    while get_epsilon(max_noise) > target_epsilon:
        min_noise, max_noise = max_noise, 2 * max_noise

    error_epsilon = lambda s: np.abs(get_epsilon(s) - target_epsilon)
    opt_result = sp_opt.minimize_scalar(
        error_epsilon,
        bounds=(min_noise, max_noise),
        method='bounded',
        options={'xatol': tol},
    )
    assert opt_result.success

    return opt_result.x


def compute_epsilon(noise_multipliers: float, batch_sizes: int, num_steps: int, num_examples: int,
                    target_delta: float = 1e-5):
    if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')

    orders = np.array((
            list(np.linspace(1.01, 8, num=50))
            + list(range(8, 64))
            + list(np.linspace(65, 512, num=10, dtype=int))
    ))

    accountant = rdp_privacy_accountant.RdpAccountant(
        orders, NeighboringRelation.ADD_OR_REMOVE_ONE)

    for t in range(1, num_steps + 1):
        q = batch_sizes / float(num_examples)
        event = dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(noise_multipliers))
        accountant.compose(event, t)

    eps, unused_opt_order = accountant.get_epsilon_and_optimal_order(target_delta=target_delta)
    return eps


class CDPAccountant:
    def __init__(self, args):
        self.args = args
        self._dp_epsilon = args.epsilon
        self._dp_delta = args.delta
        self.batch_size = args.batch_size

        if args.clipping_norm is None:
            self._clipping_norm = float('inf')
        elif args.clipping_norm < 0:
            raise ValueError('Clipping norm must be non-negative.')
        else:
            self._clipping_norm = args.clipping_norm

        if args.noise_multiplier is None:
            self._noise_multiplier = 0
        elif args.noise_multiplier < 0:
            raise ValueError('Standard deviation must be non-negative.')
        else:
            self._noise_multiplier = args.noise_multiplier

        if args.comm_round is None:
            self._comm_round = None
        else:
            self._comm_round = args.comm_round

        self._batch_size = args.batch_size
        self._epochs = args.epochs
        self._client_num_in_total = args.client_num_in_total
        self._client_num_per_round = args.client_num_per_round
        self.stop_training_at_epsilon = args.stop_training_at_epsilon

    def finite_dp_guarantee(self) -> bool:
        """Returns whether the DP guarantee (eps, delta) can be finite."""
        # The privacy (eps, delta) can only be finite with non-zero noise
        # and with a finite clipping-norm.
        return bool(self._noise_multiplier and np.isfinite(self._clipping_norm))

    def compute_max_server_steps(self):
        """Compute maximum number of updates given the DP parameters."""
        if self.finite_dp_guarantee():
            return calibrate_steps(target_epsilon=self._dp_epsilon, target_delta=self._dp_delta,
                                   noise_multipliers=self._noise_multiplier, batch_sizes=self._batch_size,
                                   num_examples=self.args.client_num_per_round)
        else:
            return 0

    def compute_max_comm_rounds(self):
        if self._comm_round is None:
            max_server_steps = self.compute_max_server_steps()
            self._comm_round = max_server_steps
            return max_server_steps
        else:
            return self._comm_round

    def compute_current_epsilon(self, cur_rounds: int) -> float:
        """Compute DP epsilon given the DP parameters and current `num_updates`."""

        return self.compute_epsilon(num_rounds=cur_rounds, noise_multipliers=self._noise_multiplier)

    def compute_total_epsilon(self):
        if not self.stop_training_at_epsilon:
            num_rounds = self.compute_max_server_steps()
            return self.compute_epsilon(num_rounds=num_rounds, noise_multipliers=self._noise_multiplier)
        else:
            return self._dp_epsilon

    def compute_epsilon(self, noise_multipliers: float, num_rounds: int, target_delta: float = 1e-5):

        if self._client_num_in_total * target_delta > 1.:
            warnings.warn('Your delta might be too high.')

        orders = np.array((
                list(np.linspace(1.01, 8, num=50))
                + list(range(8, 64))
                + list(np.linspace(65, 512, num=10, dtype=int))
        ))

        accountant = rdp_privacy_accountant.RdpAccountant(
            orders, NeighboringRelation.ADD_OR_REMOVE_ONE)

        for t in range(1, num_rounds + 1):
            q = self._client_num_per_round / float(self._client_num_in_total)
            event = dp_event.PoissonSampledDpEvent(
                q, dp_event.GaussianDpEvent(noise_multipliers))
            accountant.compose(event, t)

        eps, unused_opt_order = accountant.get_epsilon_and_optimal_order(target_delta=target_delta)
        return eps


class LDPAccountant:

    def __init__(self, args):
        self.args = args
        self._dp_epsilon = args.epsilon
        self._dp_delta = args.delta
        self._batch_size = args.batch_size

        if args.clipping_norm is None:
            self._clipping_norm = float('inf')
        elif args.clipping_norm < 0:
            raise ValueError('Clipping norm must be non-negative.')
        else:
            self._clipping_norm = args.clipping_norm

        if args.noise_multiplier is None:
            self._noise_multiplier = 0
        elif args.noise_multiplier < 0:
            raise ValueError('Standard deviation must be non-negative.')
        else:
            self._noise_multiplier = args.noise_multiplier

        self._batch_size = args.batch_size
        self._epochs = args.epochs
        self.stop_training_at_epsilon = args.stop_training_at_epsilon

    def finite_dp_guarantee(self) -> bool:
        """Returns whether the DP guarantee (eps, delta) can be finite."""
        # The privacy (eps, delta) can only be finite with non-zero noise
        # and with a finite clipping-norm.
        return bool(self._noise_multiplier and np.isfinite(self._clipping_norm))

    def compute_max_local_num_steps(self, num_samples):
        """only when all clients participate"""
        if self.finite_dp_guarantee():
            return calibrate_steps(
                target_epsilon=self._dp_epsilon,
                noise_multipliers=self._noise_multiplier,
                batch_sizes=self._batch_size,
                num_examples=num_samples,
                target_delta=self._dp_delta,
            )
        else:
            return 0

    def compute_current_epsilon(self, num_steps, num_samples) -> float:
        return compute_epsilon(num_steps=num_steps, noise_multipliers=self._noise_multiplier,
                               batch_sizes=self._batch_size, num_examples=num_samples)

    def compute_max_comm_rounds(self, train_data_num):
        steps_per_epoch = train_data_num / self._batch_size
        return np.ceil(
            self.compute_max_local_num_steps(num_samples=train_data_num) / (self._epochs * steps_per_epoch))

    def compute_total_epsilon(self, train_data_num):
        if not self.stop_training_at_epsilon:
            num_steps = self._epochs * int(train_data_num / self._batch_size)
            return compute_epsilon(num_steps=num_steps, noise_multipliers=self._noise_multiplier,
                                   batch_sizes=self._batch_size, num_examples=train_data_num)
        else:
            return self._dp_epsilon
