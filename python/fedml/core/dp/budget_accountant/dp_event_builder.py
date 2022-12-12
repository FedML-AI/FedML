"""Builder class for ComposedDpEvent."""

from fedml.core.dp.budget_accountant import dp_event


class DpEventBuilder(object):
    """Constructs a `DpEvent` representing the composition of a series of events.

    Two common use cases of the `DpEventBuilder` are 1) for producing and tracking
    a ledger of `DpEvent`s during sequential accounting using a
    `PrivacyAccountant`, and 2) for building up a description of a composite
    mechanism for subsequent batch accounting.
    """

    def __init__(self):
        # A list of (event, count) pairs.
        self._event_counts = []
        self._composed_event = None

    def compose(self, event: dp_event.DpEvent, count: int = 1):
        """Composes new event into event represented by builder.

        Args:
          event: The new event to compose.
          count: The number of times to compose the event.
        """
        if not isinstance(event, dp_event.DpEvent):
            raise TypeError('`event` must be a subclass of `DpEvent`. '
                            f'Found {type(event)}.')
        if not isinstance(count, int):
            raise TypeError(f'`count` must be an integer. Found {type(count)}.')
        if count < 1:
            raise ValueError(f'`count` must be positive. Found {count}.')

        if isinstance(event, dp_event.NoOpDpEvent):
            return
        elif isinstance(event, dp_event.SelfComposedDpEvent):
            self.compose(event.event, count * event.count)
        else:
            if self._event_counts and self._event_counts[-1][0] == event:
                new_event_count = (event, self._event_counts[-1][1] + count)
                self._event_counts[-1] = new_event_count
            else:
                self._event_counts.append((event, count))
            self._composed_event = None

    def build(self) -> dp_event.DpEvent:
        """Builds and returns the composed DpEvent represented by the builder."""
        if not self._composed_event:
            events = []
            for event, count in self._event_counts:
                if count == 1:
                    events.append(event)
                else:
                    events.append(dp_event.SelfComposedDpEvent(event, count))
            if not events:
                self._composed_event = dp_event.NoOpDpEvent()
            elif len(events) == 1:
                self._composed_event = events[0]
            else:
                self._composed_event = dp_event.ComposedDpEvent(events)

        return self._composed_event
