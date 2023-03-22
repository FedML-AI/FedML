# Privacy Budget Accountant

This directory contains tools for tracking differential privacy budgets, available as part of the [Google differential privacy library](https://github.com/google/differential-privacy).

The set of DpEvent classes allow you to describe complex differentially private mechanisms such as Laplace and Gaussian, subsampling mechanisms, and their compositions. The PrivacyAccountant classes can ingest DpEvents and return the ε, δ of the composite mechanism. Privacy Loss Distributions (PLDs) and RDP accounting are currently supported.



