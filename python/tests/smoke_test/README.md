Smoke Test is usually a fast, simple check of major functionality to confirm that a build is ready to be subject to further testing. It can prevent flawed artifacts being released into QA environments, wasting both time and resources.

The current smoke test uses GitHub Actions as CI/CD tool and contains user-level scripts for following modules:
- FedML CLI: login, build, logs, logout.
- FedML Parrot (Simulation) SP using a simple and fast workload: LR + MNIST
- FedML Parrot (Simulation) MPI using a simple and fast workload: LR + MNIST
- FedML Parrot (Simulation) NCCL using a simple and fast workload: LR + MNIST
- FedML Octopus (Cross-silo) using a simple and fast workload: LR + MNIST
- FedML Beehive (Cross-device) using a simple and fast workload: LR + MNIST
