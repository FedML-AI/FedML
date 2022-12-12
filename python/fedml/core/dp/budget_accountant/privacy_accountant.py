"""PrivacyAccountant abstract base class."""

import abc
import enum

from fedml.core.dp.budget_accountant import dp_event
from fedml.core.dp.budget_accountant.dp_event_builder import DpEventBuilder


class NeighboringRelation(enum.Enum):
    ADD_OR_REMOVE_ONE = 1
    REPLACE_ONE = 2

    # A record is replaced with a special record, such as the "zero record". See
    # https://arxiv.org/pdf/2103.00039.pdf, Definition 1.1.
    REPLACE_SPECIAL = 3


class UnsupportedEventError(Exception):
    """Exception to raise if _compose is called on unsupported event type."""


class PrivacyAccountant(metaclass=abc.ABCMeta):
    """Abstract base class for privacy accountants."""

    def __init__(self, neighboring_relation: NeighboringRelation):
        self._neighboring_relation = neighboring_relation
        self._ledger = DpEventBuilder()

    @property
    def neighboring_relation(self) -> NeighboringRelation:
        """The neighboring relation used by the accountant.

        The neighboring relation is expected to remain constant after
        initialization. Subclasses should not override this property or change the
        value of the private attribute.
        """
        return self._neighboring_relation

    @abc.abstractmethod
    def supports(self, event: dp_event.DpEvent) -> bool:
        """Checks whether the `DpEvent` can be processed by this accountant.

        In general this will require recursively checking the structure of the
        `DpEvent`. In particular `ComposedDpEvent` and `SelfComposedDpEvent` should
        be recursively examined.

        Args:
          event: The `DpEvent` to check.

        Returns:
          True iff this accountant supports processing `event`.
        """

    @abc.abstractmethod
    def _compose(self, event: dp_event.DpEvent, count: int = 1):
        """Updates internal state to account for application of a `DpEvent`.

        Calls to `get_epsilon` or `get_delta` after calling `_compose` will return
        values that account for this `DpEvent`.

        Args:
          event: A `DpEvent` to process.
          count: The number of times to compose the event.
        """

    def compose(self, event: dp_event.DpEvent, count: int = 1):
        """Updates internal state to account for application of a `DpEvent`.

        Calls to `get_epsilon` or `get_delta` after calling `compose` will return
        values that account for this `DpEvent`.

        Args:
          event: A `DpEvent` to process.
          count: The number of times to compose the event.

        Raises:
          UnsupportedEventError: `event` is not supported by this
          `PrivacyAccountant`.
        """
        if not isinstance(event, dp_event.DpEvent):
            raise TypeError(f'`event` must be `DpEvent`. Found {type(event)}.')

        if not self.supports(event):
            raise UnsupportedEventError(f'Unsupported event: {event}.')

        self._ledger.compose(event, count)
        self._compose(event, count)

    @property
    def ledger(self) -> dp_event.DpEvent:
        """Returns the (composed) `DpEvent` processed so far by this accountant."""
        return self._ledger.build()

    @abc.abstractmethod
    def get_epsilon(self, target_delta: float) -> float:
        """Gets the current epsilon.

        Args:
          target_delta: The target delta.

        Returns:
          The current epsilon, accounting for all composed `DpEvent`s.
        """

    def get_delta(self, target_epsilon: float) -> float:
        """Gets the current delta.

        An implementer of `PrivacyAccountant` may choose not to override this, in
        which case `NotImplementedError` will be raised.

        Args:
          target_epsilon: The target epsilon.

        Returns:
          The current delta, accounting for all composed `DpEvent`s.
        """
        raise NotImplementedError()
