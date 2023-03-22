"""Basic Example for Using Privacy Loss Distributions.
"""

from absl import app

from dp_accounting import privacy_loss_distribution


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # The parameter of Laplace Noise added
  parameter_laplace = 3
  # PLD for one execution of the Laplace Mechanism. (Throughout we assume that
  # sensitivity = 1.)
  laplace_pld = privacy_loss_distribution.from_laplace_mechanism(
      parameter_laplace, value_discretization_interval=1e-3)

  # Number of times Laplace Mechanism is run
  num_laplace = 40
  # PLD for num_laplace executions of the Laplace Mechanism.
  composed_laplace_pld = laplace_pld.self_compose(num_laplace)

  epsilon = 10
  delta = composed_laplace_pld.get_delta_for_epsilon(epsilon)
  print(f'An algorithm that executes the Laplace Mechanism with parameter '
        f'{parameter_laplace} for a total of {num_laplace} times is '
        f'({epsilon}, {delta})-DP.')

  # PLDs for different mechanisms can also be composed. Below is an example in
  # which we compose PLDs for Laplace Mechanism and Gaussian Mechanism.

  # STD of the Gaussian Noise
  standard_deviation = 5
  # PLD for an execution of the Gaussian Mechanism.
  gaussian_pld = privacy_loss_distribution.from_gaussian_mechanism(
      standard_deviation, value_discretization_interval=1e-3)

  # PLD for num_laplace executions of the Laplace Mechanism and one execution of
  # the Gaussian Mechanism.
  composed_laplace_and_gaussian_pld = composed_laplace_pld.compose(gaussian_pld)

  epsilon = 10
  delta = composed_laplace_and_gaussian_pld.get_delta_for_epsilon(epsilon)
  print(f'An algorithm that executes the Laplace Mechanism with parameter '
        f'{parameter_laplace} for a total of {num_laplace} times and in '
        f'addition executes once the Gaussian Mechanism with STD '
        f'{standard_deviation} is ({epsilon}, {delta})-DP.')

if __name__ == '__main__':
  app.run(main)
