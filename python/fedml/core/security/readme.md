# Attack
1. ByzantineAttack: (1) zero mode (2) random mode (3) flip mode
2. (NeurIPS 2019) DLGAttack: "Deep leakage from gradients" 
https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf
3. (NeurIPS 2020) InvertAttack: "Inverting gradients-how easy is it to break privacy in federated learning?"
https://github.com/JonasGeiping/invertinggradients/
4. LabelFlippingAttack: "Data Poisoning Attacks Against Federated Learning Systems" 
https://arxiv.org/pdf/2007.08432
5. (NeurIPS 2021) RevealingLabelsFromGradientsAttack: "Revealing and Protecting Labels in Distributed Training" 
https://proceedings.neurips.cc/paper/2021/file/0d924f0e6b3fd0d91074c22727a53966-Paper.pdf
6. (NeurIPS 2019) BackdoorAttack: "A Little Is Enough: Circumventing Defenses For Distributed Learning" 
https://proceedings.neurips.cc/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf
7. (NeurIPS 2020) EdgeCaseBackdoorAttack: "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning" 
https://proceedings.neurips.cc/paper/2020/file/b8ffa41d4e492f0fad2f13e29e1762eb-Paper.pdf
8. (PMLR'20) ModelReplacementBackdoorAttack: "How To Backdoor Federated Learning" 
http://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf

**Todo**
1. (ICML 2019) "Analyzing Federated Learning through an Adversarial Lens" 
http://proceedings.mlr.press/v97/bhagoji19a/bhagoji19a.pdf
2. (ICLR 2020) "DBA: Distributed Backdoor Attacks against Federated Learning" 
https://openreview.net/pdf?id=rkgyS0VFvr
3. (USENIX 2020) "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" 
https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf
4. (ICML 2021) Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix. 
http://proceedings.mlr.press/v139/lam21b/lam21b.pdf
5. “Model poisoning attacks against distributed machine learning systems” 
https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11006/110061D/Model-poisoning-attacks-against-distributed-machine-learning-systems/10.1117/12.2520275.full?SSO=1
6. The Limitations of Federated Learning in Sybil Settings
7. The Hidden Vulnerability of Distributed Learning in Byzantium

**Maybe implementing in the future...**
1. (Vertical FL) "CAFE: Catastrophic Data Leakage in Vertical Federated Learning" (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/file/08040837089cdf46631a10aca5258e16-Paper.pdf


# Defense
1. (PMLR 2018) BulyanDefense: "The Hidden Vulnerability of Distributed Learning in Byzantium. "
http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf
2. CClipDefense: "Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
https://arxiv.org/pdf/2006.09365.pdf
3. GeometricMedianDefense: "Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. "
https://dl.acm.org/doi/pdf/10.1145/3154503
4. (NeurIPS 2017) KrumDefense: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
https://papers.nips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
5. (ICLR 2021) MultiKrumDefense: "Distributed momentum for byzantine-resilient stochastic gradient descent"
https://infoscience.epfl.ch/record/287261
6. NormDiffClippingDefense: "Can You Really Backdoor Federated Learning?" 
https://arxiv.org/pdf/1911.07963.pdf 
7. (AAAI 2021) RobustLearningRateDefense: "Defending against backdoors in federated learning with robust learning rate."
https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate
8. SoteriaDefense: "Provable defense against privacy leakage in federated learning from representation perspective." 
https://arxiv.org/pdf/2012.06043
9. SLSGDDefense: "SLSGD: Secure and efficient distributed on-device machine learning"
https://arxiv.org/pdf/1903.06996.pdf
10. RFA_defense: "Robust Aggregation for Federated Learning"
https://arxiv.org/pdf/1912.13445
11. (USENIX2020) FoolsGoldDefense: "The Limitations of Federated Learning in Sybil Settings"
https://www.usenix.org/system/files/raid20-fung.pdf
12. (ICML 2018) CoordinateWiseMedianDefense: "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
http://proceedings.mlr.press/v80/yin18a/yin18a.pdf
13. (NeurIPS 2021) WbcDefense: "Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective" 
https://arxiv.org/abs/2110.13864
14. (ICML 2021) CRFLDefense: "CRFL: Certifiably Robust Federated Learning against Backdoor Attacks"
http://proceedings.mlr.press/v139/xie21a/xie21a.pdf



**Todo**
1. "Attack-Resistant Federated Learning with Residual-based Reweighting"
https://arxiv.org/pdf/1912.11464


**Maybe to implement in the future...**
1. "RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets"
https://ojs.aaai.org/index.php/AAAI/article/view/3968
2. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping". NDSS 2021
https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/
3. (DiverseFL) "Byzantine-Resilient Federated Learning with Heterogeneous Data Distribution"
https://arxiv.org/abs/2010.07541v3
4. "DETOX: A Redundancy-based Framework for Faster and More Robust Gradient Aggregation"
https://arxiv.org/abs/1907.12205
