"""Abstract base class for tests of `PrivacyAccountant` classes.

Checks that a class derived from `PrivacyAccountant` has the correct behavior
for standard `DpEvent` classes.
"""

from typing import Collection

from absl.testing import absltest

from fedml.core.dp.budget_accountant import dp_event
from fedml.core.dp.budget_accountant import privacy_accountant


@absltest.skipThisClass('only intended to be run by subclasses')
class PrivacyAccountantTest(absltest.TestCase):

    def _make_test_accountants(
            self) -> Collection[privacy_accountant.PrivacyAccountant]:
        """Makes a list of accountants to test.

        Subclasses should define this to return a list of accountants to be tested.

        Returns:
          A list of accountants to test.
        """
        return []

    def test_make_test_accountants(self):
        self.assertNotEmpty(self._make_test_accountants())

    def test_unsupported(self):

        class UnknownDpEvent(dp_event.DpEvent):
            pass

        for accountant in self._make_test_accountants():
            for unsupported in [dp_event.UnsupportedDpEvent(), UnknownDpEvent()]:
                self.assertFalse(accountant.supports(unsupported))
                self.assertFalse(
                    accountant.supports(dp_event.SelfComposedDpEvent(unsupported, 10)))
                self.assertFalse(
                    accountant.supports(dp_event.ComposedDpEvent([unsupported])))

    def test_no_events(self):
        for accountant in self._make_test_accountants():
            self.assertEqual(accountant.get_epsilon(1e-12), 0)
            self.assertEqual(accountant.get_epsilon(0), 0)
            self.assertEqual(accountant.get_epsilon(1), 0)
            try:
                self.assertEqual(accountant.get_delta(1e-12), 0)
                self.assertEqual(accountant.get_delta(0), 0)
                self.assertEqual(accountant.get_delta(float('inf')), 0)
            except NotImplementedError:
                # Implementing `get_delta` is optional.
                pass

    def test_no_op(self):
        for accountant in self._make_test_accountants():
            event = dp_event.NoOpDpEvent()
            self.assertTrue(accountant.supports(event))
            accountant._compose(event)
            self.assertEqual(accountant.get_epsilon(1e-12), 0)
            self.assertEqual(accountant.get_epsilon(0), 0)
            self.assertEqual(accountant.get_epsilon(1), 0)
            try:
                self.assertEqual(accountant.get_delta(1e-12), 0)
                self.assertEqual(accountant.get_delta(0), 0)
                self.assertEqual(accountant.get_delta(float('inf')), 0)
            except NotImplementedError:
                # Implementing `get_delta` is optional.
                pass

    def test_non_private(self):
        for accountant in self._make_test_accountants():
            event = dp_event.NonPrivateDpEvent()
            self.assertTrue(accountant.supports(event))
            accountant._compose(event)
            self.assertEqual(accountant.get_epsilon(0.99), float('inf'))
            self.assertEqual(accountant.get_epsilon(0), float('inf'))
            self.assertEqual(accountant.get_epsilon(1), float('inf'))
            try:
                self.assertEqual(accountant.get_delta(100), 1)
                self.assertEqual(accountant.get_delta(0), 1)
                self.assertEqual(accountant.get_delta(float('inf')), 1)
            except NotImplementedError:
                # Implementing `get_delta` is optional.
                pass
