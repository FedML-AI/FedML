"""Tests for DpEventBuilder."""

from absl.testing import absltest
from fedml.core.dp.budget_accountant import dp_event
from fedml.core.dp.budget_accountant import dp_event_builder

_gaussian_event = dp_event.GaussianDpEvent(1.0)
_laplace_event = dp_event.LaplaceDpEvent(1.0)
_poisson_event = dp_event.PoissonSampledDpEvent(_gaussian_event, 0.1)
_self_composed_event = dp_event.SelfComposedDpEvent(_gaussian_event, 3)


class DpEventBuilderTest(absltest.TestCase):

  def test_no_op(self):
    builder = dp_event_builder.DpEventBuilder()
    self.assertEqual(dp_event.NoOpDpEvent(), builder.build())

  def test_single_gaussian(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_gaussian_event)
    self.assertEqual(_gaussian_event, builder.build())

  def test_single_laplace(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_laplace_event)
    self.assertEqual(_laplace_event, builder.build())

  def test_compose_no_op(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(dp_event.NoOpDpEvent())
    builder.compose(_gaussian_event)
    builder.compose(dp_event.NoOpDpEvent())
    self.assertEqual(_gaussian_event, builder.build())

  def test_compose_self(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_gaussian_event)
    builder.compose(_gaussian_event, 2)
    self.assertEqual(_self_composed_event, builder.build())

  def test_compose_heterogenous(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_poisson_event)
    builder.compose(_gaussian_event)
    builder.compose(_gaussian_event, 2)
    builder.compose(_poisson_event)
    expected_event = dp_event.ComposedDpEvent(
        [_poisson_event, _self_composed_event, _poisson_event])
    self.assertEqual(expected_event, builder.build())

  def test_compose_composed(self):
    builder = dp_event_builder.DpEventBuilder()
    composed_event = dp_event.ComposedDpEvent(
        [_gaussian_event, _poisson_event, _self_composed_event])
    builder.compose(_gaussian_event)
    builder.compose(composed_event)
    builder.compose(composed_event, 2)
    builder.compose(_poisson_event)
    builder.compose(_poisson_event)
    expected_event = dp_event.ComposedDpEvent([
        _gaussian_event,
        dp_event.SelfComposedDpEvent(composed_event, 3),
        dp_event.SelfComposedDpEvent(_poisson_event, 2)
    ])
    self.assertEqual(expected_event, builder.build())


if __name__ == '__main__':
  absltest.main()
