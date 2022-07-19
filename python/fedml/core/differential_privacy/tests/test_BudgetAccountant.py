from unittest import TestCase

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.utils import Budget, BudgetError


class TestBudgetAccountant(TestCase):
    def tearDown(self):
        BudgetAccountant.pop_default()

    @staticmethod
    def sample_model(epsilon=1.0, accountant=None):
        accountant = BudgetAccountant.load_default(accountant)
        accountant.check(epsilon, 0.0)

        accountant.spend(epsilon, 0.0)

    @staticmethod
    def sample_model2(epsilon=1.0, accountant=None):
        accountant = BudgetAccountant.load_default(accountant)
        accountant.check(epsilon, 0.0)

        accountant.spend(epsilon, 0.0)

    def test_init(self):
        acc = BudgetAccountant()
        self.assertEqual(acc.epsilon, float("inf"))
        self.assertEqual(acc.delta, 1)

    def test_init_epsilon(self):
        acc = BudgetAccountant(1, 0)
        self.assertEqual(acc.epsilon, 1.0)
        self.assertEqual(acc.delta, 0.0)

    def test_init_delta(self):
        acc = BudgetAccountant(0, 0.5)
        self.assertEqual(acc.epsilon, 0)
        self.assertAlmostEqual(acc.delta, 0.5, places=5)

    def test_init_neg_eps(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(-1)

    def test_init_neg_del(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, -0.5)

    def test_init_large_del(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, 1.5)

    def test_init_zero_budget(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(0, 0)

    def test_init_scalar_spent(self):
        with self.assertRaises(TypeError):
            BudgetAccountant(spent_budget=2)

    def test_init_non_list_spent(self):
        with self.assertRaises(TypeError):
            BudgetAccountant(spent_budget=(1, 0))

    def test_init_small_tuple_spent(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(spent_budget=[(1,)])

    def test_init_large_tuple_spent(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(spent_budget=[(1, 0, 2)])

    def test_init_large_slack(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, 1e-2, 1e-1)

    def test_change_large_slack(self):
        acc = BudgetAccountant(1, 0.2, 0)
        acc.spend(0.1, 0.1)
        acc.spend(0.1, 0.1)

        with self.assertRaises(BudgetError):
            acc.slack = 0.2

    def test_total_large_slack(self):
        acc = BudgetAccountant(1, 0.5)
        with self.assertRaises(ValueError):
            acc.total(slack=0.75)

    def test_init_exceed_budget(self):
        with self.assertRaises(BudgetError):
            BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0), (0.5, 0)])

    def test_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(BudgetError):
            acc.check(0.1, 0)

    def test_get_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        spent_budget = acc.spent_budget

        self.assertIsInstance(spent_budget, list)
        self.assertEqual(2, len(spent_budget))

    def test_change_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(AttributeError):
            acc.spent_budget = [(1, 0)]

        with self.assertRaises(AttributeError):
            del acc.spent_budget

        acc.spent_budget.append((1, 0))
        self.assertEqual(2, len(acc))
        self.assertEqual(acc.spent_budget, acc._BudgetAccountant__spent_budget)
        self.assertIsNot(acc.spent_budget, acc._BudgetAccountant__spent_budget)

    def test_change_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(AttributeError):
            acc.epsilon = 2

        with self.assertRaises(AttributeError):
            del acc.epsilon

        with self.assertRaises(AttributeError):
            acc.delta = 0.1

        with self.assertRaises(AttributeError):
            del acc.delta

    def test_get_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        self.assertEqual(1, acc.epsilon)
        self.assertEqual(0, acc.delta)

    def test_remaining_budget_epsilon(self):
        acc = BudgetAccountant(1, 0)
        eps, delt = acc.remaining()
        self.assertAlmostEqual(eps, 1.0)
        self.assertEqual(delt, 0)

        acc = BudgetAccountant(1, 0)
        eps, delt = acc.remaining(10)
        self.assertAlmostEqual(eps, 0.1)
        self.assertEqual(delt, 0)

    def test_remaining_budget_epsilon_slack(self):
        acc = BudgetAccountant(1, 1e-15, slack=1e-15)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(eps, 0.01)
        self.assertEqual(delt, 0)

    def test_remaining_budget_delta(self):
        acc = BudgetAccountant(1, 1e-2)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(delt, 1e-4)
        self.assertLessEqual(delt, 1e-3)

        acc = BudgetAccountant(1, 1e-2, slack=1e-5)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(eps, 0.01)
        self.assertGreaterEqual(delt, 1e-4)
        self.assertLessEqual(delt, 1e-3)

    def test_remaining_budget_zero_delta(self):
        acc = BudgetAccountant(1, 1e-2, 1e-2)
        _, delt = acc.remaining(100)
        self.assertEqual(0.0, delt)

    def test_remaining_budget_k(self):
        acc = BudgetAccountant(1, 1e-2, 1e-3)

        with self.assertRaises(ValueError):
            acc.remaining(0)

        with self.assertRaises(TypeError):
            acc.remaining(1.0)

    def test_remaining_budget_inf(self):
        acc = BudgetAccountant()
        self.assertEqual((float("inf"), 1.0), acc.remaining())
        self.assertEqual((float("inf"), 1.0), acc.remaining(100))

        acc.spend(float("inf"), 1)
        self.assertEqual((float("inf"), 1.0), acc.remaining())
        self.assertEqual((float("inf"), 1.0), acc.remaining(100))

    def test_spend(self):
        acc = BudgetAccountant()
        acc.spend(1, 0)
        self.assertEqual(acc.total(), (1, 0))

        acc.spend(1, 0.5)
        self.assertEqual(acc.total(), (2, 0.5))

        acc.spend(5, 0)
        self.assertEqual(acc.total(), (7, 0.5))

    def test_spend_errors(self):
        acc = BudgetAccountant()

        with self.assertRaises(ValueError):
            acc.spend(0, 0)

        with self.assertRaises(ValueError):
            acc.spend(-1, 0)

        with self.assertRaises(ValueError):
            acc.spend(1, -1)

        with self.assertRaises(ValueError):
            acc.spend(1, 2)

    def test_spend_exceed(self):
        acc = BudgetAccountant(5, 0)
        acc.spend(3, 0)

        with self.assertRaises(BudgetError):
            acc.spend(3, 0)

        with self.assertRaises(BudgetError):
            acc.spend(0, 1e-5)

    def test_inf_spend(self):
        acc = BudgetAccountant()
        acc.spend(float("inf"), 1)
        self.assertEqual((float("inf"), 1), acc.total())
        self.assertEqual((float("inf"), 1), acc.remaining())
        self.assertEqual((float("inf"), 1), acc.remaining(100))
        self.assertTrue(acc.check(float("inf"), 1))

    def test_remaining_budget_positive_vals(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5, [(0.01, 0), (0.01, 0), (0.01, 0)])
        eps, delt = acc.remaining(50)
        self.assertGreaterEqual(eps, 0)
        self.assertGreaterEqual(delt, 0)

    def test_remaining_budget_implementation(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5, [(0.01, 0), (0.01, 0), (0.01, 0)])
        k = 50

        eps, delt = acc.remaining(k)

        for i in range(k-1):
            acc.spend(eps, delt)

        remaining_eps, remaining_delt = acc.remaining()

        self.assertAlmostEqual(remaining_eps, eps)
        self.assertAlmostEqual(remaining_delt, delt)

    def test_remaining_budget_implementation2(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5)
        k = 50

        eps, delt = acc.remaining(k)

        for i in range(k//2):
            acc.spend(eps, delt)

        eps, delt = acc.remaining(k)

        for i in range(k-1):
            acc.spend(eps, delt)

        remaining_eps, remaining_delt = acc.remaining()

        self.assertAlmostEqual(remaining_eps, eps)
        self.assertAlmostEqual(remaining_delt, delt)

    def test_load_wrong_type(self):
        with self.assertRaises(TypeError):
            BudgetAccountant.load_default(0)

        with self.assertRaises(TypeError):
            BudgetAccountant.load_default([1, 2, 3])

        with self.assertRaises(TypeError):
            BudgetAccountant.load_default("BudgetAccountant")

    def test_small_epsilon(self):
        acc = BudgetAccountant(1)

        with self.assertRaises(ValueError):
            acc.spend(1e-16, 0)

    def test_set_default(self):
        acc = BudgetAccountant()
        acc.set_default()

        self.assertIs(BudgetAccountant._default, acc)

    def test_pop_default(self):
        acc = BudgetAccountant().set_default()
        acc2 = BudgetAccountant.pop_default()

        self.assertIs(acc, acc2)

    def test_default(self):
        # Specify accountant as arg
        acc1 = BudgetAccountant(1.5, 0.0)
        self.sample_model(accountant=acc1)
        self.assertEqual((1.0, 0.0), acc1.total())

        # Use default accountant without one being set
        self.sample_model()
        acc2 = BudgetAccountant.pop_default()
        self.assertIsNot(acc1, acc2)
        self.assertEqual(float("inf"), acc2.epsilon)
        self.assertEqual((1.0, 0.0), acc2.total())

        # Set accountant as default
        acc3 = BudgetAccountant(2.0, 0.0).set_default()
        self.sample_model(epsilon=1.5)
        self.assertEqual((1.5, 0), acc3.total())
        self.assertEqual(2.0, acc3.epsilon)
        self.assertIsNot(acc3, acc2)
        self.assertIsNot(acc3, acc1)

        # Check default is same as what we set it
        acc4 = BudgetAccountant.pop_default()
        self.assertIs(acc3, acc4)
        self.assertEqual((1.0, 0.0), acc2.total())

        # Run again in 2 different functions without setting a default
        self.sample_model()
        self.sample_model2()
        acc5 = BudgetAccountant.pop_default()
        self.assertIsNot(acc5, acc2)
        self.assertIsNot(acc5, acc3)
        self.assertEqual((2.0, 0), acc5.total())

    def test_correct_composition(self):
        epsilons = [2**-9] * 700
        slack = 2**-25

        acc = BudgetAccountant(slack=slack)

        for epsilon in epsilons:
            acc.spend(epsilon, 0)

        spent_epsilon, spent_delta = acc.total()

        self.assertAlmostEqual(spent_epsilon, 0.27832280615743646366122002955588987576913442137093, places=14)
        self.assertEqual(spent_delta, slack)

    def test_with_statement(self):
        acc = BudgetAccountant()
        acc2 = BudgetAccountant()

        with acc:
            self.sample_model(1)
            self.sample_model(2)

        self.assertIsNone(BudgetAccountant._default)
        self.assertEqual((3, 0), acc.total())

        acc.set_default()

        with acc2:
            self.sample_model(5)
            self.sample_model(1, accountant=acc)

        self.assertEqual((4, 0), acc.total())
        self.assertEqual((5, 0), acc2.total())
        self.assertIs(BudgetAccountant._default, acc)

    def test_with_statement_errors(self):
        with BudgetAccountant(1.5) as acc:
            self.assertIsInstance(acc, BudgetAccountant)

            self.sample_model(1)

            with self.assertRaises(BudgetError):
                self.sample_model(1)

        with self.assertRaises(BudgetError):
            with BudgetAccountant(1):
                self.sample_model(2)

    def test_budget_output(self):
        acc = BudgetAccountant()
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

        acc.spend(1, 0.5)
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

        acc.slack = 0.5
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

    def test_len(self):
        acc = BudgetAccountant()
        self.assertEqual(0, len(acc))

        acc.spend(1, 0)
        acc.spend(1, 0)
        self.assertEqual(2, len(acc))

        acc.spend(1, 0)
        self.assertEqual(3, len(acc))

    def test_repr(self):
        acc = BudgetAccountant()

        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertEqual("BudgetAccountant()", repr(acc))

        acc = BudgetAccountant(epsilon=1, delta=0.01, slack=0.01)
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertIn("epsilon", repr(acc))
        self.assertIn("delta", repr(acc))
        self.assertIn("slack", repr(acc))
        self.assertNotIn("spent_budget", repr(acc))

        acc = BudgetAccountant(spent_budget=[(1, 0), (0, 1)])
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertNotIn("epsilon", repr(acc))
        self.assertNotIn("delta", repr(acc))
        self.assertNotIn("slack", repr(acc))
        self.assertIn("spent_budget", repr(acc))

        acc = BudgetAccountant(spent_budget=[(1., 0.)] * 10 + [(5., 0.5)])
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertIn("...", repr(acc))
        self.assertNotIn("5", repr(acc))
        self.assertNotIn("5", acc.__repr__(10))
        self.assertIn("5", acc.__repr__(11))
