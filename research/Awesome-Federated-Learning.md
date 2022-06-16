## FedML - The Most Popular Federated Learning Library https://fedml.ai

# Awesome-Federated-Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of federated learning publications, re-organized from Arxiv (mostly).

<strong>Last Update: July, 20th, 2021</strong>.	

If your publication is not included here, please email to chaoyanghe.com@gmail.com

# Foundations and Trends in Machine Learning
We are thrilled to share that [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) has been accepted to [FnTML](https://www.nowpublishers.com/MAL) (<b>Foundations and Trends in Machine Learning</b>, the chief editor is [Michael Jordan](https://people.eecs.berkeley.edu/~jordan/)).

[A Field Guide to Federated Optimization](https://arxiv.org/abs/2107.06917)


## Publications in Top-tier ML/CV/NLP/DM Conference (ICML, NeurIPS, ICLR, CVPR, ACL, AAAI, KDD)
### ICML
| Title                                                                    | Team/Authors              | Venue and Year     | Targeting Problem     | Method                |
|---|---|---|---|---|
| [Federated Learning with Only Positive Labels](https://arxiv.org/pdf/2004.10342.pdf)                        | Google Research            |   ICML 2020        | label deficiency in multi-class classification    |  regularization |
| [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)        | EPFL, Google Research      |   ICML 2020        | heterogeneous data (non-I.I.D)    | nonconvex/convex optimization with variance reduction   |
| [FedBoost: A Communication-Efficient Algorithm for Federated Learning](https://proceedings.icml.cc/static/paper_files/icml/2020/5967-Paper.pdf)    | Google Research, NYU       |   ICML 2020        | communication cost    | ensemble algorithm    |
| [FetchSGD: Communication-Efficient Federated Learning with Sketching](https://arxiv.org/abs/2007.07682)     | UC Berkeley, JHU, Amazon   |   ICML 2020        | communication cost    | compress model updates with Count Sketch   |
| [From Local SGD to Local Fixed-Point Methods for Federated Learning](https://arxiv.org/pdf/2004.01442.pdf)  | KAUST                      |   ICML 2020        | communication cost    |  Optimization |

### NeurIPS
| Title                                                                    | Team/Authors              | Venue and Year     | Targeting Problem     | Method                |
|---|---|---|---|---|
| Lower Bounds and Optimal Algorithms for Personalized Federated Learning | KAUST   | NeurIPS 2020        |  non-I.I.D, personalization   |   |
| Personalized Federated Learning with Moreau Envelopes  | The University of Sydney | NeurIPS 2020        |  non-I.I.D, personalization   |   |
| Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach | MIT |   NeurIPS 2020        |  non-I.I.D, personalization   |   |
| Differentially-Private Federated Contextual Bandits                     | MIT            |   NeurIPS 2020        |  Contextual Bandits   |   |
| Federated Principal Component Analysis                     | Cambridge            |   NeurIPS 2020        |    PCA |   |
| FedSplit: an algorithmic framework for fast federated optimization                     | UCB            |   NeurIPS 2020        |   Acceleration  |   |
| Federated Bayesian Optimization via Thompson Sampling | MIT |   NeurIPS 2020        |     |   |
| Robust Federated Learning: The Case of Affine Distribution Shifts | MIT  | NeurIPS 2020        |   Privacy, Robustness  |   |
| An Efficient Framework for Clustered Federated Learning | UCB | NeurIPS 2020        |    heterogeneous data (non-I.I.D) |   |
| Distributionally Robust Federated Averaging | PSU |   NeurIPS 2020        |  Privacy, Robustness   |   |
| Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge | USC |   NeurIPS 2020        |  Efficient Training of Large DNN at Edge   |   |
| A Scalable Approach for Privacy-Preserving Collaborative Machine Learning  | USC |   NeurIPS 2020        |  Scalability   |   |
| Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization | CMU |   NeurIPS 2020        |   local update step heterogeneity  |   |
| Attack of the Tails: Yes, You Really Can Backdoor Federated Learning | Wiscosin|   NeurIPS 2020        |  Privacy, Robustness   |   |
| Federated Accelerated Stochastic Gradient Descent | Stanford |   NeurIPS 2020        |  Acceleration   |   |
| Inverting Gradients - How easy is it to break privacy in federated learning? | University of Siegen   | NeurIPS 2020        |  Privacy, Robustness   |   |
| Ensemble Distillation for Robust Model Fusion in Federated Learning  | EPFL |   NeurIPS 2020        |   Privacy, Robustness  |   |
| Optimal Topology Design for Cross-Silo Federated Learning  | Inria | NeurIPS 2020        | Topology Optimization    |   |
| Distributed Training with Heterogeneous Data: Bridging Median- and Mean-Based Algorithms | University of Minnesota | NeurIPS 2020 | | | 
| Distributed Distillation for On-Device Learning | Stanford | NeurIPS 2020 | | |
| Byzantine Resilient Distributed Multi-Task Learning | Vanderbilt University | NeurIPS 2020 | | | 
| Distributed Newton Can Communicate Less and Resist Byzantine Workers | UCB | NeurIPS 2020 | | |
| Minibatch vs Local SGD for Heterogeneous Distributed Learning | TTIC | NeurIPS 2020 | | |
| Election Coding for Distributed Learning: Protecting SignSGD against Byzantine Attacks | | NeurIPS 2020 | | |

(according to https://neurips.cc/Conferences/2020/AcceptedPapersInitial)

Note: most of the accepted publications are preparing the camera ready revision, thus we are not sure the detail of their proposed methods


## Research Areas
#### Statistical Challenges: data distribution heterogeneity and label deficiency (159)
 - [Distributed Optimization](#Distributed-optimization (68))
 - [Non-IID and Model Personalization](#Non-IID-and-Model-Personalization (53))
 - [Semi-Supervised Learning](#Semi-Supervised-Learning (3))
 - [Vertical Federated Learning](#Vertical-Federated-Learning (8))
 - [Decentralized FL](#Decentralized-FL (7))
 - [Hierarchical FL](#Hierarchical-FL (8))
 - [Neural Architecture Search](#Neural-Architecture-Search (4))
 - [Transfer Learning](#Transfer-Learning (11))
 - [Continual Learning](#continual-learning (1))
 - [Domain Adaptation](#Domain-Adaptation)
 - [Reinforcement Learning](#Reinforcement-Learning)
 - [Bayesian Learning ](#Bayesian-Learning )
 - [Causal Learning](#Causal-Learning )


#### Trustworthiness: security, privacy, fairness, incentive mechanism, etc. (88)
 - [Adversarial-Attack-and-Defense](#Adversarial-Attack-and-Defense)
 - [Privacy](#Privacy (36))
 - [Fairness](#Fairness (4))
 - [Interpretability](#Interpretability)
 - [Incentive Mechanism](#Incentive-Mechanism (5))

#### System Challenges: communication and computational resource constrained, software and hardware heterogeneity, and FL system (141)
 - [Communication-Efficiency](#Communication-Efficiency (29))
 - [Straggler Problem](#straggler-problem (4))
 - [Computation Efficiency](#Computation-Efficiency (14))
 - [Wireless Communication and Cloud Computing](#Wireless-Communication-and-Cloud-Computing (74))
 - [FL System Design](#FL-System-Design (20))

#### Models and Applications (104)
 - [Models](#Models (22))
 - [Natural language Processing](#Natural-language-Processing (15))
 - [Computer Vision](#Computer-Vision (3))
 - [Health Care](#Health-Care (27))
 - [Transportation](#Transportation (14))
 - [Recommendation System](#Recommendation-System (8))
 - [Speech](#Speech (1))
 - [Finance](#Finance (2))
 - [Smart City](#Smart-City (2))
 - [Robotics](#Robotics (2))
 - [Networking](#Networking (1))
 - [Blockchain](#Blockchain (2))
 - [Other](#Other (5))
 
#### Benchmark, Dataset and Survey (27)
 - [Benchmark and Dataset](#Benchmark-and-Dataset)  (7)
 - [Survey](#Survey) (20)

-------------------

# Statistical Challenges: distribution heterogeneity and label deficiency 

## Distributed optimization
<span style="color:blue">Userful Federated Optimizer Baselines:</span>

FedAvg:
[Communication-Efficient Learning of Deep Networks from Decentralized Data. 2016-02. AISTAT 2017.](https://arxiv.org/pdf/1602.05629.pdf)

FedOpt:
[Adaptive Federated Optimization. ICLR 2021 (Under Review). 2020-02-29](https://arxiv.org/pdf/2003.00295.pdf)

FedNov:
[Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. NeurIPS 2020](https://arxiv.org/abs/2007.07481)

-------------------------

[Federated Optimization: Distributed Optimization Beyond the Datacenter. NIPS 2016 workshop.](https://arxiv.org/pdf/1511.03575.pdf)

[Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/pdf/1610.02527.pdf)

[Stochastic, Distributed and Federated Optimization for Machine Learning. FL PhD Thesis. By Jakub](https://arxiv.org/pdf/1707.01155.pdf)

[Collaborative Deep Learning in Fixed Topology Networks](https://arxiv.org/pdf/1706.07880.pdf)

[Federated Multi-Task Learning](https://arxiv.org/pdf/1705.10467.pdf)

[LAG: Lazily Aggregated Gradient for Communication-Efficient Distributed Learning](https://arxiv.org/abs/1805.09965)

[Local Stochastic Approximation: A Unified View of Federated Learning and Distributed Multi-Task Reinforcement Learning Algorithms](https://arxiv.org/pdf/2006.13460.pdf)

[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning](https://arxiv.org/pdf/2005.06105.pdf)

[Exact Support Recovery in Federated Regression with One-shot Communication](https://arxiv.org/pdf/2006.12583.pdf)

[DEED: A General Quantization Scheme for Communication Efficiency in Bits](https://arxiv.org/pdf/2006.11401.pdf)
Researcher: Ruoyu Sun, UIUC

[Robust Federated Learning: The Case of Affine Distribution Shifts](https://arxiv.org/pdf/2006.08907.pdf)

[Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/pdf/2006.08848.pdf)

[Towards Flexible Device Participation in Federated Learning for Non-IID Data](https://arxiv.org/pdf/2006.06954.pdf)
Keywords: inactive or return incomplete updates in non-IID dataset

[A Primal-Dual SGD Algorithm for Distributed Nonconvex Optimization](https://arxiv.org/pdf/2006.03474.pdf)

[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](https://arxiv.org/pdf/2005.11418.pdf)
Researcher: Wotao Yin, UCLA

[FedSplit: An algorithmic framework for fast federated optimization](https://arxiv.org/pdf/2005.05238.pdf)

[Distributed Stochastic Non-Convex Optimization: Momentum-Based Variance Reduction](https://arxiv.org/pdf/2005.00224.pdf)

[On the Outsized Importance of Learning Rates in Local Update Methods](https://arxiv.org/pdf/2007.00878.pdf)
Highlight: local model learning rate optimization + automation
Researcher: Jakub

[Federated Learning with Compression: Unified Analysis and Sharp Guarantees](https://arxiv.org/pdf/2007.01154.pdf)
Highlight: non-IID, gradient compression + local SGD
Researcher: Mehrdad Mahdavi, Jin Rong’s PhD Student http://www.cse.psu.edu/~mzm616/

[From Local SGD to Local Fixed-Point Methods for Federated Learning](https://arxiv.org/pdf/2004.01442.pdf)

[Federated Residual Learning. 2020-03](https://arxiv.org/pdf/2003.12880.pdf)


[Acceleration for Compressed Gradient Descent in Distributed and Federated Optimization. ICML 2020.](https://arxiv.org/pdf/2002.11364.pdf)

[LASG: Lazily Aggregated Stochastic Gradients for Communication-Efficient Distributed Learning](https://arxiv.org/pdf/2002.11360.pdf)

[Uncertainty Principle for Communication Compression in Distributed and Federated Learning and the Search for an Optimal Compressor](https://arxiv.org/pdf/2002.08958.pdf)

[Dynamic Federated Learning](https://arxiv.org/pdf/2002.08782.pdf)

[Distributed Optimization over Block-Cyclic Data](https://arxiv.org/pdf/2002.07454.pdf)

[Distributed Non-Convex Optimization with Sublinear Speedup under Intermittent Client Availability](https://arxiv.org/pdf/2002.07399.pdf)

[Federated Learning with Matched Averaging](https://arxiv.org/pdf/2002.06440.pdf)

[Federated Learning of a Mixture of Global and Local Models](https://arxiv.org/pdf/2002.05516.pdf)

[Faster On-Device Training Using New Federated Momentum Algorithm](https://arxiv.org/pdf/2002.02090.pdf)

[FedDANE: A Federated Newton-Type Method](https://arxiv.org/pdf/2001.01920.pdf)

[Distributed Fixed Point Methods with Compressed Iterates](https://arxiv.org/pdf/1912.09925.pdf)

[Primal-dual methods for large-scale and distributed convex optimization and data analytics](https://arxiv.org/pdf/1912.08546.pdf)

[Parallel Restarted SPIDER - Communication Efficient Distributed Nonconvex Optimization with Optimal Computation Complexity](https://arxiv.org/pdf/1912.06036.pdf)

[Representation of Federated Learning via Worst-Case Robust Optimization Theory](https://arxiv.org/pdf/1912.05571.pdf)

[On the Convergence of Local Descent Methods in Federated Learning](https://arxiv.org/pdf/1910.14425.pdf)

[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)

[Central Server Free Federated Learning over Single-sided Trust Social Networks](https://arxiv.org/pdf/1910.04956.pdf)

[Accelerating Federated Learning via Momentum Gradient Descent](https://arxiv.org/pdf/1910.03197.pdf)

[Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction](https://arxiv.org/pdf/1909.05844.pdf)

[Gradient Descent with Compressed Iterates](https://arxiv.org/pdf/1909.04716.pdf)

[First Analysis of Local GD on Heterogeneous Data](https://arxiv.org/pdf/1909.04715.pdf)

[(*) On the Convergence of FedAvg on Non-IID Data. ICLR 2020.](https://arxiv.org/pdf/1907.02189.pdf)

[Robust Federated Learning in a Heterogeneous Environment](https://arxiv.org/pdf/1906.06629.pdf)

[Scalable and Differentially Private Distributed Aggregation in the Shuffled Model](https://arxiv.org/pdf/1906.08320.pdf)

[Variational Federated Multi-Task Learning](https://arxiv.org/pdf/1906.06268.pdf)

[Bayesian Nonparametric Federated Learning of Neural Networks. ICLR 2019.](https://arxiv.org/pdf/1905.12022.pdf)

[Differentially Private Learning with Adaptive Clipping](https://arxiv.org/pdf/1905.03871.pdf)

[Semi-Cyclic Stochastic Gradient Descent](https://arxiv.org/pdf/1904.10120.pdf)

[Asynchronous Federated Optimization](https://arxiv.org/pdf/1903.03934.pdf)

[Agnostic Federated Learning](https://arxiv.org/pdf/1902.00146.pdf)

[Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

[Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/pdf/1811.11206.pdf)

[Learning Rate Adaptation for Federated and Differentially Private Learning](https://arxiv.org/pdf/1809.03832.pdf)

[Communication-Efficient Robust Federated Learning Over Heterogeneous Datasets](https://arxiv.org/pdf/2006.09992.pdf)

[An Efficient Framework for Clustered Federated Learning](https://arxiv.org/pdf/2006.04088.pdf)

[Adaptive Federated Learning in Resource Constrained Edge Computing Systems](https://arxiv.org/pdf/1804.05271.pdf)
Citation: 146

[Adaptive Federated Optimization](http://arxiv.org/pdf/2003.00295.pdf)

[Local SGD converges fast and communicates little](https://arxiv.org/pdf/1805.09767.pdf)

[Don’t Use Large Mini-Batches, Use Local SGD](https://arxiv.org/pdf/1808.07217.pdf)

[Overlap Local-SGD: An Algorithmic Approach to Hide Communication Delays in Distributed SGD](https://arxiv.org/pdf/2002.09539.pdf)

[Local SGD With a Communication Overhead Depending Only on the Number of Workers](https://arxiv.org/pdf/2006.02582.pdf)

[Federated Accelerated Stochastic Gradient Descent ](https://arxiv.org/pdf/2006.08950.pdf)

[Tighter Theory for Local SGD on Identical and Heterogeneous Data](https://arxiv.org/pdf/1909.04746.pdf)

[STL-SGD: Speeding Up Local SGD with Stagewise Communication Period](https://arxiv.org/pdf/2006.06377.pdf)

[Cooperative SGD: A unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms](https://arxiv.org/pdf/1808.07576.pdf)

[Don't Use Large Mini-Batches, Use Local SGD](https://arxiv.org/pdf/1808.07217.pdf)

[Understanding Unintended Memorization in Federated Learning](http://arxiv.org/pdf/2006.07490.pdf)

## Non-IID and Model Personalization
[The Non-IID Data Quagmire of Decentralized Machine Learning. 2019-10](https://arxiv.org/pdf/1910.00189.pdf)

[Federated Learning with Non-IID Data](https://arxiv.org/pdf/1806.00582.pdf)

[FedCD: Improving Performance in non-IID Federated Learning. 2020](https://arxiv.org/pdf/2006.09637.pdf)

[Life Long Learning: FedFMC: Sequential Efficient Federated Learning on Non-iid Data. 2020](https://arxiv.org/pdf/2006.10937.pdf)

[Robust Federated Learning: The Case of Affine Distribution Shifts. 2020](https://arxiv.org/pdf/2006.08907.pdf)

[Personalized Federated Learning with Moreau Envelopes. 2020](https://arxiv.org/pdf/2006.08848.pdf)


[Personalized Federated Learning using Hypernetworks. 2021](https://arxiv.org/pdf/2103.04628.pdf)

[Ensemble Distillation for Robust Model Fusion in Federated Learning. 2020](https://arxiv.org/pdf/2006.07242.pdf)
Researcher: Tao Lin, ZJU, EPFL https://tlin-tao-lin.github.io/index.html

[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning. 2020](https://arxiv.org/pdf/2005.06105.pdf)

[Towards Flexible Device Participation in Federated Learning for Non-IID Data. 2020](https://arxiv.org/pdf/2006.06954.pdf)
Keywords: inactive or return incomplete updates in non-IID dataset

[XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning. 2020](https://arxiv.org/pdf/2006.05148.pdf)

[NeurIPS 2020 submission: An Efficient Framework for Clustered Federated Learning. 2020](https://arxiv.org/pdf/2006.04088.pdf)
Researcher: AVISHEK GHOSH, UCB, PhD

[Continual Local Training for Better Initialization of Federated Models. 2020](https://arxiv.org/pdf/2005.12657.pdf)

[FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data. 2020](https://arxiv.org/pdf/2005.11418.pdf)
Researcher: Wotao Yin, UCLA

[Global Multiclass Classification from Heterogeneous Local Models. 2020](https://arxiv.org/pdf/2005.10848.pdf)
Researcher: Stanford https://stanford.edu/~pilanci/

[Multi-Center Federated Learning. 2020](https://arxiv.org/pdf/2005.01026.pdf)

[Federated learning with hierarchical clustering of local updates to improve training on non-IID data. 2020](https://arxiv.org/pdf/2004.11791.pdf)

[Federated Learning with Only Positive Labels. 2020](https://arxiv.org/pdf/2004.10342.pdf)
Researcher: Felix Xinnan Yu, Google New York
Keywords: positive labels
Limited Labels

[Federated Semi-Supervised Learning with Inter-Client Consistency. 2020](https://arxiv.org/pdf/2006.12097.pdf)

[(*) FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning. CMU ECE. 2020-04-07](https://arxiv.org/pdf/2004.03657.pdf)

[(*) Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461.pdf)

[Semi-Federated Learning](https://arxiv.org/pdf/2003.12795.pdf)

[Survey of Personalization Techniques for Federated Learning. 2020-03-19](https://arxiv.org/pdf/2003.08673.pdf)

[Device Heterogeneity in Federated Learning: A Superquantile Approach. 2020-02](https://arxiv.org/pdf/2002.11223.pdf)

[Personalized Federated Learning for Intelligent IoT Applications: A Cloud-Edge based Framework](https://arxiv.org/pdf/2002.10671.pdf)

[Three Approaches for Personalization with Applications to Federated Learning](https://arxiv.org/pdf/2002.10619.pdf)

[Personalized Federated Learning: A Meta-Learning Approach](https://arxiv.org/pdf/2002.07948.pdf)

[Towards Federated Learning: Robustness Analytics to Data Heterogeneity](https://arxiv.org/pdf/2002.05038.pdf)
Highlight: non-IID + adversarial attacks

[Salvaging Federated Learning by Local Adaptation](https://arxiv.org/pdf/2002.04758.pdf)
Highlight: an experimental paper that evaluate FL can help to improve the local accuracy

[FOCUS: Dealing with Label Quality Disparity in Federated Learning. 2020-01](https://arxiv.org/pdf/2001.11359.pdf)

[Overcoming Noisy and Irrelevant Data in Federated Learning. ICPR 2020.](https://arxiv.org/pdf/2001.08300.pdf)

[Real-Time Edge Intelligence in the Making: A Collaborative Learning Framework via Federated Meta-Learning. 2020-01](https://arxiv.org/pdf/2001.03229.pdf)

[(*) Think Locally, Act Globally: Federated Learning with Local and Global Representations. NeurIPS 2019 Workshop on Federated Learning distinguished student paper award](https://arxiv.org/pdf/2001.01523.pdf)

[Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818.pdf)

[Federated Adversarial Domain Adaptation](https://arxiv.org/pdf/1911.02054.pdf)

[Federated Evaluation of On-device Personalization](https://arxiv.org/pdf/1910.10252.pdf)

[Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating](https://arxiv.org/pdf/1910.08234.pdf)

[Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/pdf/1910.07796.pdf)

[Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/pdf/1910.01991.pdf)

[Robust and Communication-Efficient Federated Learning From Non-i.i.d. Data](https://arxiv.org/pdf/1903.02891.pdf)

[Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488.pdf)

[Measure Contribution of Participants in Federated Learning](https://arxiv.org/pdf/1909.08525.pdf)

[(*) Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335.pdf)

[Multi-hop Federated Private Data Augmentation with Sample Compression](https://arxiv.org/pdf/1907.06426.pdf)

[Astraea: Self-balancing Federated Learning for Improving Classification Accuracy of Mobile Deep Learning Applications](https://arxiv.org/pdf/1907.01132.pdf)

[Distributed Training with Heterogeneous Data: Bridging Median- and Mean-Based Algorithms](https://arxiv.org/pdf/1906.01736.pdf)

[Hybrid-FL for Wireless Networks: Cooperative Learning Mechanism Using Non-IID Data](https://arxiv.org/pdf/1905.07210.pdf)

[Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891.pdf)

[High Dimensional Restrictive Federated Model Selection with multi-objective Bayesian Optimization over shifted distributions](https://arxiv.org/pdf/1902.08999.pdf)

[Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge](https://arxiv.org/pdf/1804.08333.pdf)

[Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/pdf/1802.07876.pdf)

[Robust Federated Learning Through Representation Matching and Adaptive Hyper-parameters](https://arxiv.org/pdf/1912.13075.pdf)

[Towards Efficient Scheduling of Federated Mobile Devices under Computational and Statistical Heterogeneity](https://arxiv.org/pdf/2005.12326.pdf)

[Client Adaptation improves Federated Learning with Simulated Non-IID Clients](https://arxiv.org/pdf/2007.04806.pdf)

[Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481.pdf)

[Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity. ICDCS 2021.](https://arxiv.org/abs/2105.00562)


## Vertical Federated Learning
[SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/pdf/1901.08755.pdf)

[Parallel Distributed Logistic Regression for Vertical Federated Learning without Third-Party Coordinator](https://arxiv.org/pdf/1911.09824.pdf)

[A Quasi-Newton Method Based Vertical Federated Learning Framework for Logistic Regression](https://arxiv.org/pdf/1912.00513.pdf)

[Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption](https://arxiv.org/pdf/1711.10677.pdf)

[Entity Resolution and Federated Learning get a Federated Resolution.](https://arxiv.org/pdf/1803.04035.pdf)

[Multi-Participant Multi-Class Vertical Federated Learning](https://arxiv.org/pdf/2001.11154.pdf)

[A Communication-Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/pdf/1912.11187.pdf)

[Asymmetrical Vertical Federated Learning](https://arxiv.org/pdf/2004.07427.pdf)
Researcher: Tencent Cloud, Libin Wang

[VAFL: a Method of Vertical Asynchronous Federated Learning, ICML workshop on FL, 2020](https://arxiv.org/abs/2007.06081)


## Decentralized FL
[Central Server Free Federated Learning over Single-sided Trust Social Networks](https://arxiv.org/pdf/1910.04956.pdf)

[Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent](https://arxiv.org/pdf/1705.09056.pdf)

[Multi-consensus Decentralized Accelerated Gradient Descent](https://arxiv.org/pdf/2005.00797.pdf)

[Decentralized Bayesian Learning over Graphs. 2019-05](https://arxiv.org/pdf/1905.10466.pdf)

[BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning](https://arxiv.org/pdf/1905.06731.pdf)

[Biscotti: A Ledger for Private and Secure Peer-to-Peer Machine Learning](https://arxiv.org/pdf/1811.09904.pdf)

[Matcha: Speeding Up Decentralized SGD via Matching Decomposition Sampling](https://arxiv.org/pdf/1905.09435.pdf)


## Hierarchical FL
[Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/pdf/1905.06641.pdf)

[(FL startup: Tongdun, HangZhou, China) Knowledge Federation: A Unified and Hierarchical Privacy-Preserving AI Framework. 2020-02](https://arxiv.org/pdf/2002.01647.pdf)

[HFEL: Joint Edge Association and Resource Allocation for Cost-Efficient Hierarchical Federated Edge Learning](https://arxiv.org/pdf/2002.11343.pdf)

[Hierarchical Federated Learning Across Heterogeneous Cellular Networks](https://arxiv.org/pdf/1909.02362.pdf)

[Enhancing Privacy via Hierarchical Federated Learning](https://arxiv.org/pdf/2004.11361.pdf)

[Federated learning with hierarchical clustering of local updates to improve training on non-IID data. 2020](https://arxiv.org/pdf/2004.11791.pdf)

[Federated Hierarchical Hybrid Networks for Clickbait Detection](https://arxiv.org/pdf/1906.00638.pdf)

[Matcha: Speeding Up Decentralized SGD via Matching Decomposition Sampling](https://arxiv.org/pdf/1905.09435.pdf) (in above section as well)

## Neural Architecture Search
[FedNAS: Federated Deep Learning via Neural Architecture Search. CVPR 2020. 2020-04-18](https://arxiv.org/pdf/2004.08546.pdf

[Real-time Federated Evolutionary Neural Architecture Search. 2020-03](https://arxiv.org/pdf/2003.02793.pdf)

[Federated Neural Architecture Search. 2020-06-14](https://arxiv.org/pdf/2002.06352.pdf)

[Differentially-private Federated Neural Architecture Search. 2020-06](https://arxiv.org/pdf/2006.10559.pdf)

## Transfer Learning

[Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf)

[Secure Federated Transfer Learning. IEEE Intelligent Systems 2018.](https://arxiv.org/pdf/1812.03337.pdf)


[FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/pdf/1910.03581.pdf)

[Secure and Efficient Federated Transfer Learning](https://arxiv.org/pdf/1910.13271.pdf)

[Wireless Federated Distillation for Distributed Edge Learning with Heterogeneous Data](https://arxiv.org/pdf/1907.02745.pdf)


[Decentralized Differentially Private Segmentation with PATE. 2020-04](https://arxiv.org/pdf/2004.06567.pdf) \
Highlights: apply the ICLR 2017 paper "Semisupervised knowledge transfer for deep learning from private training data"

[Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning. 2020](https://arxiv.org/pdf/2005.06105.pdf)

[(FL startup: Tongdun, HangZhou, China) Knowledge Federation: A Unified and Hierarchical Privacy-Preserving AI Framework. 2020-02](https://arxiv.org/pdf/2002.01647.pdf)

[Cooperative Learning via Federated Distillation over Fading Channels](https://arxiv.org/pdf/2002.01337.pdf)


[(*) Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer](https://arxiv.org/pdf/1912.11279.pdf)

[Federated Reinforcement Distillation with Proxy Experience Memory](https://arxiv.org/pdf/1907.06536.pdf)

## Continual Learning
[Federated Continual Learning with Adaptive Parameter Communication. 2020-03](https://arxiv.org/pdf/2003.03196.pdf)

## Semi-Supervised Learning
[Federated Semi-Supervised Learning with Inter-Client Consistency. 2020](https://arxiv.org/pdf/2006.12097.pdf)

[Semi-supervised knowledge transfer for deep learning from private training data. ICLR 2017](https://arxiv.org/pdf/1610.05755.pdf)

[Scalable private learning with PATE. ICLR 2018. ](https://arxiv.org/pdf/1802.08908.pdf)


## Domain Adaptation
[Federated Adversarial Domain Adaptation. ICLR 2020.](https://arxiv.org/pdf/1911.02054.pdf)

## Reinforcement Learning
[Federated Deep Reinforcement Learning](https://arxiv.org/pdf/1901.08277.pdf)

## Bayesian Learning 
[Differentially Private Federated Variational Inference. NeurIPS 2019 FL Workshop. 2019-11-24.](https://arxiv.org/pdf/1911.10563.pdf)

## Causal Learning
[Towards Causal Federated Learning For Enhanced Robustness and Privacy. ICLR 2021 DPML Workshop](https://arxiv.org/pdf/2104.06557.pdf)

# Trustworthy AI: adversarial attack, privacy, fairness, incentive mechanism, etc.

## Adversarial Attack and Defense
[An Overview of Federated Deep Learning Privacy Attacks and Defensive Strategies. 2020-04-01](https://arxiv.org/pdf/2004.04676.pdf)
Citation: 0

[How To Backdoor Federated Learning. 2018-07-02. AISTATS 2020](https://arxiv.org/pdf/1807.00459.pdf)
Citation: 128

[Can You Really Backdoor Federated Learning?. NeruIPS 2019. 2019-11-18](https://arxiv.org/pdf/1911.07963.pdf)
Highlight: by Google
Citation: 9

[DBA: Distributed Backdoor Attacks against Federated Learning. ICLR 2020.](https://openreview.net/pdf?id=rkgyS0VFvr)
Citation: 66

[CRFL: Certifiably Robust Federated Learning against Backdoor Attacks. ICML 2021.](https://arxiv.org/pdf/2106.08283.pdf)

[Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning. ACM CCS 2017. 2017-02-14](https://arxiv.org/pdf/1702.07464.pdf)
Citation: 284

[Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/pdf/1803.01498.pdf)
Citation: 112

[Deep Leakage from Gradients. NIPS 2019](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)
Citation: 31

[Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning. 2018-12-03](https://arxiv.org/pdf/1812.00910.pdf)
Citation: 46

[Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning. INFOCOM 2019](https://arxiv.org/pdf/1812.00535.pdf)
Citation: 56
Highlight: server-side attack

[Analyzing Federated Learning through an Adversarial Lens. ICML 2019.](https://arxiv.org/pdf/1811.12470.pdf). 
Citation: 60
Highlight: client attack

[Mitigating Sybils in Federated Learning Poisoning. 2018-08-14. RAID 2020](https://arxiv.org/pdf/1808.04866.pdf)
Citation: 41
Highlight: defense

[RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets, AAAI 2019](https://arxiv.org/abs/1811.03761)
Citation: 34

[(*) A Framework for Evaluating Gradient Leakage Attacks in Federated Learning. 2020-04-22](https://arxiv.org/pdf/2004.10397.pdf)
Researcher: Wenqi Wei, Ling Liu, GaTech

[(*) Local Model Poisoning Attacks to Byzantine-Robust Federated Learning. 2019-11-26](https://arxiv.org/pdf/1911.11815.pdf)

[NeurIPS 2020 Submission: Backdoor Attacks on Federated Meta-Learning](https://arxiv.org/pdf/2006.07026.pdf)
Researcher: Chien-Lun Chen, USC

[Towards Realistic Byzantine-Robust Federated Learning. 2020-04-10](https://arxiv.org/pdf/2004.04986.pdf)

[Data Poisoning Attacks on Federated Machine Learning. 2020-04-19](https://arxiv.org/pdf/2004.10020.pdf)

[Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning. 2020-04-27](https://arxiv.org/pdf/2004.12571.pdf)

[Byzantine-Resilient High-Dimensional SGD with Local Iterations on Heterogeneous Data. 2020-06-22](https://arxiv.org/pdf/2006.13041.pdf)
Researcher: Suhas Diggavi, UCLA (https://scholar.google.com/citations?hl=en&user=hjTzNuQAAAAJ&view_op=list_works&sortby=pubdate)

[(*) NeurIPS 2020 submission: FedMGDA+: Federated Learning meets Multi-objective Optimization. 2020-06-20](https://arxiv.org/pdf/2006.11489.pdf)

[(*) NeurIPS 2020 submission: Free-rider Attacks on Model Aggregation in Federated Learning. 2020-06-26](https://arxiv.org/pdf/2006.11901.pdf)

[FDA3 : Federated Defense Against Adversarial Attacks for Cloud-Based IIoT Applications. 2020-06-28](https://arxiv.org/pdf/2006.15632.pdf)


[Privacy-preserving Weighted Federated Learning within Oracle-Aided MPC Framework. 2020-05-17](https://arxiv.org/pdf/2003.07630.pdf)
Citation: 0

[BASGD: Buffered Asynchronous SGD for Byzantine Learning. 2020-03-02](https://arxiv.org/pdf/2003.00937.pdf)

[Stochastic-Sign SGD for Federated Learning with Theoretical Guarantees. 2020-02-25](https://arxiv.org/pdf/2002.10940.pdf)
Citation: 1

[Learning to Detect Malicious Clients for Robust Federated Learning. 2020-02-01](https://arxiv.org/pdf/2002.00211.pdf)

[Robust Aggregation for Federated Learning. 2019-12-31](https://arxiv.org/pdf/1912.13445.pdf)
Citation: 9

[Towards Deep Federated Defenses Against Malware in Cloud Ecosystems. 2019-12-27](https://arxiv.org/pdf/1912.12370.pdf)

[Attack-Resistant Federated Learning with Residual-based Reweighting. 2019-12-23](https://arxiv.org/pdf/1912.11464.pdf)

[Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer. 2019-12-24](https://arxiv.org/pdf/1912.11279.pdf)
Citation: 1

[Free-riders in Federated Learning: Attacks and Defenses. 2019-11-28](https://arxiv.org/pdf/1911.12560.pdf)

[Robust Federated Learning with Noisy Communication. 2019-11-01](https://arxiv.org/pdf/1911.00251.pdf)
Citation: 4

[Abnormal Client Behavior Detection in Federated Learning. 2019-10-22](https://arxiv.org/pdf/1910.09933.pdf)
Citation: 3

[Eavesdrop the Composition Proportion of Training Labels in Federated Learning. 2019-10-14](https://arxiv.org/pdf/1910.06044.pdf)
Citation: 0

[Byzantine-Robust Federated Machine Learning through Adaptive Model Averaging. 2019-09-11](https://arxiv.org/pdf/1909.05125.pdf)

[An End-to-End Encrypted Neural Network for Gradient Updates Transmission in Federated Learning. 2019-08-22](https://arxiv.org/pdf/1908.08340.pdf)

[Secure Distributed On-Device Learning Networks With Byzantine Adversaries. 2019-06-03](https://arxiv.org/pdf/1906.00887.pdf)
Citation: 3

[Robust Federated Training via Collaborative Machine Teaching using Trusted Instances. 2019-05-03](https://arxiv.org/pdf/1905.02941.pdf)
Citation: 2

[Dancing in the Dark: Private Multi-Party Machine Learning in an Untrusted Setting. 2018-11-23](https://arxiv.org/pdf/1811.09712.pdf)
Citation: 4

[Inverting Gradients - How easy is it to break privacy in federated learning? 2020-03-31](https://arxiv.org/pdf/2003.14053.pdf)
Citation: 3

[Quantification of the Leakage in Federated Learning. 2019-10-12](https://arxiv.org/pdf/1910.05467.pdf)
Citation: 1

## Privacy
[Practical Secure Aggregation for Federated Learning on User-Held Data. NIPS 2016 workshop](https://arxiv.org/pdf/1611.04482.pdf)
Highlight: cryptology

[Differentially Private Federated Learning: A Client Level Perspective. NIPS 2017 Workshop](https://arxiv.org/pdf/1712.07557.pdf)

[Exploiting Unintended Feature Leakage in Collaborative Learning. S&P 2019. 2018-05-10](https://arxiv.org/pdf/1805.04049.pdf)
Citation: 105

[(x) Gradient-Leaks: Understanding and Controlling Deanonymization in Federated Learning. 2018-05](https://arxiv.org/pdf/1805.05838.pdf)

[A Hybrid Approach to Privacy-Preserving Federated Learning. AISec 2019. 2018-12-07](https://arxiv.org/pdf/1812.03224.pdf)
Citation: 35

[A generic framework for privacy preserving deep learning. PPML 2018. 2018-11-09](https://arxiv.org/pdf/1811.04017.pdf)
Citation: 36

[Federated Generative Privacy. IJCAI 2019 FL workshop. 2019-10-08](https://arxiv.org/pdf/1910.08385.pdf)
Citation: 4

[Enhancing the Privacy of Federated Learning with Sketching. 2019-11-05](https://arxiv.org/pdf/1911.01812.pdf)
Citaiton: 0

[Federated Learning with Bayesian Differential Privacy. 2019-11-22](https://arxiv.org/pdf/1911.10071.pdf)
Citation: 5

HybridAlpha: An Efficient Approach for Privacy-Preserving Federated Learning. AISec 2019. 2019-12-12
[https://aisec.cc/](https://arxiv.org/pdf/1912.05897.pdf)

[Private Federated Learning with Domain Adaptation. NeurIPS 2019 FL workshop. 2019-12-13](https://arxiv.org/pdf/1912.06733.pdf)

[iDLG: Improved Deep Leakage from Gradients. 2020-01-08](https://arxiv.org/pdf/2001.02610.pdf)
Citation: 3

[Anonymizing Data for Privacy-Preserving Federated Learning. 2020-02-21](https://arxiv.org/pdf/2002.09096.pdf)

[Practical and Bilateral Privacy-preserving Federated Learning. 2020-02-23](https://arxiv.org/pdf/2002.09843.pdf)
Citation: 0

[Decentralized Policy-Based Private Analytics. 2020-03-14](https://arxiv.org/pdf/2003.06612.pdf)
Citation: 0

[FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection. DASFAA 2020. 2020-03-24](https://arxiv.org/pdf/2003.10637.pdf)
Citation: 0

[Learn to Forget: User-Level Memorization Elimination in Federated Learning. 2020-03-24](https://arxiv.org/pdf/2003.10933.pdf)

[LDP-Fed: Federated Learning with Local Differential Privacy. EdgeSys 2020. 2020-04-01](https://arxiv.org/pdf/2006.03637.pdf)
Researcher: Ling Liu, GaTech
Citation: 1

[PrivFL: Practical Privacy-preserving Federated Regressions on High-dimensional Data over Mobile Networks. 2020-04-05](https://arxiv.org/pdf/2004.02264.pdf)
Citation: 0

[Local Differential Privacy based Federated Learning for Internet of Things. 2020-04-09](https://arxiv.org/pdf/2004.08856.pdf)
Citation: 0

[Differentially Private AirComp Federated Learning with Power Adaptation Harnessing Receiver Noise. 2020-04.](https://arxiv.org/pdf/2004.06337.pdf)

[Decentralized Differentially Private Segmentation with PATE. MICCAI 2020 Under Review. 2020-04](https://arxiv.org/pdf/2004.06567.pdf) \
Highlights: apply the ICLR 2017 paper "Semisupervised knowledge transfer for deep learning from private training data"


[Enhancing Privacy via Hierarchical Federated Learning. 2020-04-23](https://arxiv.org/pdf/2004.11361.pdf)

[Privacy Preserving Distributed Machine Learning with Federated Learning. 2020-04-25](https://arxiv.org/pdf/2004.12108.pdf)
Citation: 0

[Exploring Private Federated Learning with Laplacian Smoothing. 2020-05-01](https://arxiv.org/pdf/2005.00218.pdf)
Citation: 0

[Information-Theoretic Bounds on the Generalization Error and Privacy Leakage in Federated Learning. 2020-05-05](https://arxiv.org/pdf/2005.02503.pdf)
Citation: 0

[Efficient Privacy Preserving Edge Computing Framework for Image Classification. 2020-05-10](https://arxiv.org/pdf/2005.04563.pdf)
Citation: 0

[A Distributed Trust Framework for Privacy-Preserving Machine Learning. 2020-06-03](https://arxiv.org/pdf/2006.02456.pdf)
Citation: 0

[Secure Byzantine-Robust Machine Learning. 2020-06-08](https://arxiv.org/pdf/2006.04747.pdf)

[ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing. 2020-06-08](https://arxiv.org/pdf/2006.04593.pdf)

[Privacy For Free: Wireless Federated Learning Via Uncoded Transmission With Adaptive Power Control. 2020-06-09](https://arxiv.org/pdf/2006.05459.pdf)
Citation: 0

[(*) Distributed Differentially Private Averaging with Improved Utility and Robustness to Malicious Parties. 2020-06-12](https://arxiv.org/pdf/2006.07218.pdf)
Citation: 0

[GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators. 2020-06-15](https://arxiv.org/pdf/2006.08848.pdf)
Citation: 0

[Federated Learning with Differential Privacy:Algorithms and Performance Analysis](https://arxiv.org/pdf/1911.00222.pdf)
Citation: 2

## Fairness
[Fair Resource Allocation in Federated Learning. ICLR 2020.](https://arxiv.org/pdf/1905.10497.pdf)

[Hierarchically Fair Federated Learning](https://arxiv.org/pdf/2004.10386.pdf)

[Towards Fair and Privacy-Preserving Federated Deep Models](https://arxiv.org/pdf/1906.01167.pdf)

## Interpretability
[Interpret Federated Learning with Shapley Values. ](https://arxiv.org/pdf/1905.04519.pdf)


## Incentive Mechanism

[Record and reward federated learning contributions with blockchain. IEEE CyberC 2019](https://mblocklab.com/RecordandReward.pdf)

[FMore: An Incentive Scheme of Multi-dimensional Auction for Federated Learning in MEC. ICDCS 2020](https://arxiv.org/pdf/2002.09699.pdf)

[Toward an Automated Auction Framework for Wireless Federated Learning Services Market](https://arxiv.org/pdf/1912.06370.pdf)

[Federated Learning for Edge Networks: Resource Optimization and Incentive Mechanism](https://arxiv.org/pdf/1911.05642.pdf)

[Motivating Workers in Federated Learning: A Stackelberg Game Perspective](https://arxiv.org/pdf/1908.03092.pdf)

[Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach](https://arxiv.org/pdf/1905.07479.pdf)

[A Learning-based Incentive Mechanism forFederated Learning](https://www.u-aizu.ac.jp/~pengli/files/fl_incentive_iot.pdf)

[A Crowdsourcing Framework for On-Device Federated Learning](https://arxiv.org/pdf/1911.01046.pdf)

# System Challenges: communication and computational resource constrained, software and hardware heterogeneity, and FL wireless communication system

## Communication Efficiency
[Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf)
Highlights: optimization

[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training. ICLR 2018. 2017-12-05](https://arxiv.org/pdf/1712.01887.pdf)
Highlights: gradient compression
Citation: 298

[NeurIPS 2020 submission: Artemis: tight convergence guarantees for bidirectional compression in Federated Learning. 2020-06-25](https://arxiv.org/pdf/2006.14591.pdf)
Highlights: bidirectional gradient compression

[Scheduling Policy and Power Allocation for Federated Learning in NOMA Based MEC. 2020-06-21](https://arxiv.org/pdf/2006.13044.pdf)

[(x) Federated Mutual Learning. 2020-06-27](https://arxiv.org/pdf/2006.16765.pdf)
Highlights: Duplicate to Deep Mutual Learning. CVPR 2018

[A Better Alternative to Error Feedback for Communication-Efficient Distributed Learning. 2020-06-19](https://arxiv.org/pdf/2006.11077.pdf)
Researcher: Peter Richtárik

[Federated Learning With Quantized Global Model Updates. 2020-06-18](https://arxiv.org/pdf/2006.10672.pdf)
Researcher: Mohammad Mohammadi Amiri, Princeton, Information Theory and Machine Learning
Highlights: model compression

[Federated Learning with Compression: Unified Analysis and Sharp Guarantees. 2020-07-02](https://arxiv.org/pdf/2007.01154.pdf)
Highlight: non-IID, gradient compression + local SGD
Researcher: Mehrdad Mahdavi, Jin Rong’s PhD http://www.cse.psu.edu/~mzm616/

[Evaluating the Communication Efficiency in Federated Learning Algorithm. 2020-04-06](https://arxiv.org/pdf/2004.02738.pdf)

[Dynamic Sampling and Selective Masking for Communication-Efficient Federated Learning. 2020-05-21](https://arxiv.org/pdf/2003.09603.pdf)

[Ternary Compression for Communication-Efficient Federated Learning. 2020-05-07](https://arxiv.org/pdf/2003.03564.pdf)

[Gradient Statistics Aware Power Control for Over-the-Air Federated Learning. 2020-05-04](https://arxiv.org/pdf/2003.02089.pdf)

[Communication-Efficient Decentralized Learning with Sparsification and Adaptive Peer Selection. 2020-02-22](https://arxiv.org/pdf/2002.09692.pdf)

[(*) RPN: A Residual Pooling Network for Efficient Federated Learning. ECAI 2020.](https://arxiv.org/pdf/2001.08600.pdf)

[Intermittent Pulling with Local Compensation for Communication-Efficient Federated Learning. 2020-01-22](https://arxiv.org/pdf/2001.08277.pdf)

[Hyper-Sphere Quantization: Communication-Efficient SGD for Federated Learning. 2019-11-12](https://arxiv.org/pdf/1911.04655.pdf)

[L-FGADMM: Layer-Wise Federated Group ADMM for Communication Efficient Decentralized Deep Learning](https://arxiv.org/pdf/1911.03654.pdf)

[Gradient Sparification for Asynchronous Distributed Training. 2019-10-24](https://arxiv.org/pdf/1910.10929.pdf)

[High-Dimensional Stochastic Gradient Quantization for Communication-Efficient Edge Learning](https://arxiv.org/pdf/1910.03865.pdf)

[SAFA: a Semi-Asynchronous Protocol for Fast Federated Learning with Low Overhead](https://arxiv.org/pdf/1910.01355.pdf)

[Detailed comparison of communication efficiency of split learning and federated learning](https://arxiv.org/pdf/1909.09145.pdf)

[Decentralized Federated Learning: A Segmented Gossip Approach](https://arxiv.org/pdf/1908.07782.pdf)

[Communication-Efficient Federated Deep Learning with Asynchronous Model Update and Temporally Weighted Aggregation](https://arxiv.org/pdf/1903.07424.pdf)

[One-Shot Federated Learning](https://arxiv.org/pdf/1902.11175.pdf)

[Multi-objective Evolutionary Federated Learning](https://arxiv.org/pdf/1812.07478.pdf)

[Expanding the Reach of Federated Learning by Reducing Client Resource Requirements](https://arxiv.org/pdf/1812.07210.pdf)

[Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/pdf/1811.11206.pdf)

[FedOpt: Towards communication efficiency and privacy preservation in federated learning](https://res.mdpi.com/d_attachment/applsci/applsci-10-02864/article_deploy/applsci-10-02864.pdf)

[A performance evaluation of federated learning algorithms](https://www.researchgate.net/profile/Gregor_Ulm/[publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms]/(links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf))



## Straggler Problem

[Coded Federated Learning. Presented at the Wireless Edge Intelligence Workshop, IEEE GLOBECOM 2019](https://arxiv.org/pdf/2002.09574.pdf)

[Turbo-Aggregate: Breaking the Quadratic Aggregation Barrier in Secure Federated Learning](https://arxiv.org/pdf/2002.04156.pdf)

[Coded Federated Computing in Wireless Networks with Straggling Devices and Imperfect CSI](https://arxiv.org/pdf/1901.05239.pdf)

[Information-Theoretic Perspective of Federated Learning](https://arxiv.org/pdf/1911.07652.pdf)


## Computation Efficiency
[NeurIPS 2020 Submission: Distributed Learning on Heterogeneous Resource-Constrained Devices](https://arxiv.org/pdf/2006.05403.pdf)

[SplitFed: When Federated Learning Meets Split Learning](https://arxiv.org/pdf/2004.12088.pdf)

[Lottery Hypothesis based Unsupervised Pre-training for Model Compression in Federated Learning](https://arxiv.org/pdf/2004.09817.pdf)

[Secure Federated Learning in 5G Mobile Networks. 2020/04](https://arxiv.org/pdf/2004.06700.pdf) 

[ELFISH: Resource-Aware Federated Learning on Heterogeneous Edge Devices](https://arxiv.org/pdf/1912.01684.pdf)

[Asynchronous Online Federated Learning for Edge Devices](https://arxiv.org/pdf/1911.02134.pdf)

[(*) Secure Federated Submodel Learning](https://arxiv.org/pdf/1911.02254.pdf)

[Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence](https://arxiv.org/pdf/1910.09594.pdf)

[Model Pruning Enables Efficient Federated Learning on Edge Devices](https://arxiv.org/pdf/1909.12326.pdf)

[Towards Effective Device-Aware Federated Learning](https://arxiv.org/pdf/1908.07420.pdf)

[Accelerating DNN Training in Wireless Federated Edge Learning System](https://arxiv.org/pdf/1905.09712.pdf)

[Split learning for health: Distributed deep learning without sharing raw patient data](https://arxiv.org/pdf/1812.00564.pdf)

[SmartPC: Hierarchical pace control in real-time federated learning system](https://www.ece.ucf.edu/~zsguo/pubs/conference_workshop/RTSS2019b.pdf)

[DeCaf: Iterative collaborative processing over the edge](https://www.usenix.org/system/files/hotedge19-paper-kumar.pdf)

## Wireless Communication and Cloud Computing
Researcher: 
H. Vincent Poor
https://ee.princeton.edu/people/h-vincent-poor

Hao Ye
https://scholar.google.ca/citations?user=ok7OWEAAAAAJ&hl=en

Ye Li
http://liye.ece.gatech.edu/

[Mix2FLD: Downlink Federated Learning After Uplink Federated Distillation With Two-Way Mixup](https://arxiv.org/pdf/2006.09801.pdf)
Researcher: Mehdi Bennis, Seong-Lyun Kim

[Wireless Communications for Collaborative Federated Learning in the Internet of Things](https://arxiv.org/pdf/2006.02499.pdf)

[Democratizing the Edge: A Pervasive Edge Computing Framework](https://arxiv.org/pdf/2007.00641.pdf)

[UVeQFed: Universal Vector Quantization for Federated Learning](https://arxiv.org/pdf/2006.03262.pdf)

[Federated Deep Learning Framework For Hybrid Beamforming in mm-Wave Massive MIMO](https://arxiv.org/pdf/2005.09969.pdf)

[Efficient Federated Learning over Multiple Access Channel with Differential Privacy Constraints](https://arxiv.org/pdf/2005.07776.pdf)

[A Secure Federated Learning Framework for 5G Networks](https://arxiv.org/pdf/2005.05752.pdf)

[Federated Learning and Wireless Communications](https://arxiv.org/pdf/2005.05265.pdf)

[Lightwave Power Transfer for Federated Learning-based Wireless Networks](https://arxiv.org/pdf/2005.03977.pdf)

[Towards Ubiquitous AI in 6G with Federated Learning](https://arxiv.org/pdf/2004.13563.pdf)

[Optimizing Over-the-Air Computation in IRS-Aided C-RAN Systems](https://arxiv.org/pdf/2004.09168.pdf)

[Network-Aware Optimization of Distributed Learning for Fog Computing](https://arxiv.org/pdf/2004.08488.pdf)

[On the Design of Communication Efficient Federated Learning over Wireless Networks](https://arxiv.org/pdf/2004.07351.pdf)

[Federated Machine Learning for Intelligent IoT via Reconfigurable Intelligent Surface](https://arxiv.org/pdf/2004.05843.pdf)

[Client Selection and Bandwidth Allocation in Wireless Federated Learning Networks: A Long-Term Perspective](https://arxiv.org/pdf/2004.04314.pdf)

[Resource Management for Blockchain-enabled Federated Learning: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/2004.04104.pdf)

[A Blockchain-based Decentralized Federated Learning Framework with Committee Consensus](https://arxiv.org/pdf/2004.00773.pdf)

[Scheduling for Cellular Federated Edge Learning with Importance and Channel. 2020-04](https://arxiv.org/pdf/2004.00490.pdf)

[Differentially Private Federated Learning for Resource-Constrained Internet of Things. 2020-03](https://arxiv.org/pdf/2003.12705.pdf)

[Federated Learning for Task and Resource Allocation in Wireless High Altitude Balloon Networks. 2020-03](https://arxiv.org/pdf/2003.09375.pdf)

[Gradient Estimation for Federated Learning over Massive MIMO Communication Systems](https://arxiv.org/pdf/2003.08059.pdf)

[Adaptive Federated Learning With Gradient Compression in Uplink NOMA](https://arxiv.org/pdf/2003.01344.pdf)

[Performance Analysis and Optimization in Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2003.00229.pdf)

[Energy-Efficient Federated Edge Learning with Joint Communication and Computation Design](https://arxiv.org/pdf/2003.00199.pdf)

[Federated Over-the-Air Subspace Learning and Tracking from Incomplete Data](https://arxiv.org/pdf/2002.12873.pdf)

[Decentralized Federated Learning via SGD over Wireless D2D Networks](https://arxiv.org/pdf/2002.12507.pdf)

[HFEL: Joint Edge Association and Resource Allocation for Cost-Efficient Hierarchical Federated Edge Learning](https://arxiv.org/pdf/2002.11343.pdf)

[Federated Learning in the Sky: Joint Power Allocation and Scheduling with UAV Swarms](https://arxiv.org/pdf/2002.08196.pdf)

[Wireless Federated Learning with Local Differential Privacy](https://arxiv.org/pdf/2002.05151.pdf)

[Cooperative Learning via Federated Distillation over Fading Channels](https://arxiv.org/pdf/2002.01337.pdf)

[Federated Learning under Channel Uncertainty: Joint Client Scheduling and Resource Allocation. 2020-02](https://arxiv.org/pdf/2002.01337.pdf)

[Learning from Peers at the Wireless Edge](https://arxiv.org/pdf/2001.11567.pdf)

[Convergence of Update Aware Device Scheduling for Federated Learning at the Wireless Edge](https://arxiv.org/pdf/2001.10402.pdf)

[Communication Efficient Federated Learning over Multiple Access Channels](https://arxiv.org/pdf/2001.08737.pdf)

[Convergence Time Optimization for Federated Learning over Wireless Networks](https://arxiv.org/pdf/2001.07845.pdf)

[One-Bit Over-the-Air Aggregation for Communication-Efficient Federated Edge Learning: Design and Convergence Analysis](https://arxiv.org/pdf/2001.05713.pdf)

[Federated Learning with Cooperating Devices: A Consensus Approach for Massive IoT Networks. IEEE Internet of Things Journal. 2020](https://arxiv.org/pdf/1912.13163.pdf)

[Asynchronous Federated Learning with Differential Privacy for Edge Intelligence](https://arxiv.org/pdf/1912.07902.pdf)

[Federated learning with multichannel ALOHA](https://arxiv.org/pdf/1912.06273.pdf)

[Federated Learning with Autotuned Communication-Efficient Secure Aggregation](https://arxiv.org/pdf/1912.00131.pdf)

[Bandwidth Slicing to Boost Federated Learning in Edge Computing](https://arxiv.org/pdf/1911.07615.pdf)

[Energy Efficient Federated Learning Over Wireless Communication Networks](https://arxiv.org/pdf/1911.02417.pdf)

[Device Scheduling with Fast Convergence for Wireless Federated Learning](https://arxiv.org/pdf/1911.00856.pdf)

[Energy-Aware Analog Aggregation for Federated Learning with Redundant Data](https://arxiv.org/pdf/1911.00188.pdf)

[Age-Based Scheduling Policy for Federated Learning in Mobile Edge Networks](https://arxiv.org/pdf/1910.14648.pdf)

[Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation](https://arxiv.org/pdf/1910.13067.pdf)

[Federated Learning over Wireless Networks: Optimization Model Design and Analysis](http://networking.khu.ac.kr/layouts/net/publications/data/2019\)Federated%20Learning%20over%20Wireless%20Network.pdf)

[Resource Allocation in Mobility-Aware Federated Learning Networks: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1910.09172.pdf)

[Reliable Federated Learning for Mobile Networks](https://arxiv.org/pdf/1910.06837.pdf)

[FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](https://arxiv.org/pdf/1909.13014.pdf)

[Active Federated Learning](https://arxiv.org/pdf/1909.12641.pdf)

[Cell-Free Massive MIMO for Wireless Federated Learning](https://arxiv.org/pdf/1909.12567.pdf)

[A Joint Learning and Communications Framework for Federated Learning over Wireless Networks](https://arxiv.org/pdf/1909.07972.pdf)

[On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.org/pdf/1909.06512.pdf)

[On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.org/pdf/1909.06512.pdf)

[Hierarchical Federated Learning Across Heterogeneous Cellular Networks](https://arxiv.org/pdf/1909.02362.pdf)

[Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/pdf/1908.06847.pdf)

[Scheduling Policies for Federated Learning in Wireless Networks](https://arxiv.org/pdf/1908.06287.pdf)

[Federated Learning with Additional Mechanisms on Clients to Reduce Communication Costs](https://arxiv.org/pdf/1908.05891.pdf)

[Federated Learning over Wireless Fading Channels](https://arxiv.org/pdf/1907.09769.pdf)

[Energy-Efficient Radio Resource Allocation for Federated Edge Learning](https://arxiv.org/pdf/1907.06040.pdf)

[Mobile Edge Computing, Blockchain and Reputation-based Crowdsourcing IoT Federated Learning: A Secure, Decentralized and Privacy-preserving System](https://arxiv.org/pdf/1906.10893.pdf)

[Active Learning Solution on Distributed Edge Computing](https://arxiv.org/pdf/1906.10718.pdf)

[Fast Uplink Grant for NOMA: a Federated Learning based Approach](https://arxiv.org/pdf/1905.04519.pdf)

[Machine Learning at the Wireless Edge: Distributed Stochastic Gradient Descent Over-the-Air](https://arxiv.org/pdf/1901.00844.pdf)

[Federated Learning via Over-the-Air Computation](https://arxiv.org/pdf/1812.11750.pdf)

[Broadband Analog Aggregation for Low-Latency Federated Edge Learning](https://arxiv.org/pdf/1812.11494.pdf)

[Federated Echo State Learning for Minimizing Breaks in Presence in Wireless Virtual Reality Networks](https://arxiv.org/pdf/1812.01202.pdf)

[Joint Service Pricing and Cooperative Relay Communication for Federated Learning](https://arxiv.org/pdf/1811.12082.pdf)

[In-Edge AI: Intelligentizing Mobile Edge Computing, Caching and Communication by Federated Learning](https://arxiv.org/pdf/1809.07857.pdf)

[Asynchronous Task Allocation for Federated and Parallelized Mobile Edge Learning](https://arxiv.org/pdf/1905.01656.pdf)

[CoLearn: enabling federated learning in MUD-compliant IoT edge networks](CoLearn: enabling federated learning in MUD-compliant IoT edge networks)

## FL System Design
[Towards Federated Learning at Scale: System Design](https://arxiv.org/pdf/1902.01046.pdf)

[FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/pdf/2007.13518.pdf)

[A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/pdf/1907.09693.pdf)

[FLeet: Online Federated Learning via Staleness Awareness and Performance Prediction](https://arxiv.org/pdf/2006.07273.pdf)
Researcher: Georgios Damaskinos, MLSys, https://people.epfl.ch/georgios.damaskinos?lang=en

[Heterogeneity-Aware Federated Learning](https://arxiv.org/pdf/2006.06983.pdf)
Researcher: Mengwei Xu, PKU

Responsive Web User Interface to Recover Training Data from User Gradients in Federated Learning
https://ldp-machine-learning.herokuapp.com/

[Decentralised Learning from Independent Multi-Domain Labels for Person Re-Identification](https://arxiv.org/pdf/2006.04150.pdf)

[[startup] Industrial Federated Learning -- Requirements and System Design](https://arxiv.org/pdf/2005.06850.pdf)

[(startup) Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://arxiv.org/pdf/2007.00914.pdf)

[(FL startup: Tongdun, HangZhou, China) Knowledge Federation: A Unified and Hierarchical Privacy-Preserving AI Framework. 2020-02](https://arxiv.org/pdf/2002.01647.pdf)

[(*) TiFL: A Tier-based Federated Learning System. HPDC 2020 (High-Performance Parallel and Distributed Computing).](https://arxiv.org/pdf/2001.09249.pdf)

[FMore: An Incentive Scheme of Multi-dimensional Auction for Federated Learning in MEC. ICDCS 2020 (2020 International Conference on Distributed Computing Systems)](https://arxiv.org/pdf/2002.09699.pdf)

[Adaptive Gradient Sparsification for Efficient Federated Learning: An Online Learning Approach. ICDCS 2020 (2020 International Conference on Distributed Computing Systems)](https://arxiv.org/pdf/2001.04756.pdf)

[Quantifying the Performance of Federated Transfer Learning](https://arxiv.org/pdf/1912.12795.pdf)

[ELFISH: Resource-Aware Federated Learning on Heterogeneous Edge Devices](https://arxiv.org/pdf/1912.01684.pdf)

[Privacy is What We Care About: Experimental Investigation of Federated Learning on Edge Devices](https://arxiv.org/pdf/1911.04559.pdf)

[Substra: a framework for privacy-preserving, traceable and collaborative Machine Learning](https://arxiv.org/pdf/1910.11567.pdf)

[BAFFLE : Blockchain Based Aggregator Free Federated Learning](https://arxiv.org/pdf/1909.07452.pdf)

[Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking](https://arxiv.org/pdf/1908.01924.pdf)

[Functional Federated Learning in Erlang (ffl-erl)](https://arxiv.org/pdf/1808.08143.pdf)

[HierTrain: Fast Hierarchical Edge AI Learning With Hybrid Parallelism in Mobile-Edge-Cloud Computing](https://arxiv.org/pdf/2003.09876.pdf)


# Models and Applications

## Models
### Graph Neural Networks

[Peer-to-peer federated learning on graphs](https://arxiv.org/pdf/1901.11173)

[Towards Federated Graph Learning for Collaborative Financial Crimes Detection](https://arxiv.org/pdf/1909.12946)

[A Graph Federated Architecture with Privacy Preserving Learning](https://arxiv.org/pdf/2104.13215)

[Federated Myopic Community Detection with One-shot Communication](https://arxiv.org/pdf/2106.07255)

[Federated Dynamic GNN with Secure Aggregation](https://arxiv.org/pdf/2009.07351)

[Privacy-Preserving Graph Neural Network for Node Classification](https://arxiv.org/pdf/2005.11903)

[ASFGNN: Automated Separated-Federated Graph Neural Network](https://arxiv.org/pdf/2011.03248)

[GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs](https://arxiv.org/pdf/2012.04187)

[FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation](https://arxiv.org/pdf/2102.04925)

[FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks](https://arxiv.org/pdf/2104.07145) 

[FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search](https://arxiv.org/pdf/2104.04141)

[Cluster-driven Graph Federated Learning over Multiple Domains](https://arxiv.org/pdf/2104.14628)

[FedGL: Federated Graph Learning Framework with Global Self-Supervision](https://arxiv.org/pdf/2105.03170)

[Federated Graph Learning -- A Position Paper](https://arxiv.org/pdf/2105.11099)

[SpreadGNN: Serverless Multi-task Federated Learning for Graph Neural Networks](https://arxiv.org/pdf/2106.02743)

[Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling](https://arxiv.org/pdf/2106.05223)

[A Vertical Federated Learning Framework for Graph Convolutional Network](https://arxiv.org/pdf/2106.11593)

[Federated Graph Classification over Non-IID Graphs](https://arxiv.org/pdf/2106.13423)

[Subgraph Federated Learning with Missing Neighbor Generation](https://arxiv.org/pdf/2106.13430)

### Federated Learning on Knowledge Graphs

[FedE: Embedding Knowledge Graphs in Federated Setting](https://arxiv.org/pdf/2010.12882)

[Improving Federated Relational Data Modeling via Basis Alignment and Weight Penalty](https://arxiv.org/pdf/2011.11369)

[Federated Knowledge Graphs Embedding](https://arxiv.org/pdf/2105.07615)


### Generative Models (GAN, Bayesian Generative Models, etc)

[Discrete-Time Cox Models](https://arxiv.org/pdf/2006.08997.pdf)

[Generative Models for Effective ML on Private, Decentralized Datasets. Google. ICLR 2020](https://arxiv.org/pdf/1911.06679.pdf)
Citation: 8

[MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets. 2018-11-09](https://arxiv.org/pdf/1811.03850.pdf)

[(GAN) Federated Generative Adversarial Learning. 2020-05-07](https://arxiv.org/pdf/2005.03793.pdf)
Citation: 0

[Differentially Private Data Generative Models](https://arxiv.org/pdf/1812.02274.pdf)

[GRAFFL: Gradient-free Federated Learning of a Bayesian Generative Model](https://arxiv.org/pdf/1910.08489.pdf)

### VAE (Variational Autoencoder)

[(VAE) An On-Device Federated Learning Approach for Cooperative Anomaly Detection](https://arxiv.org/pdf/2002.12301.pdf)

### MF (Matrix Factorization)

[Secure Federated Matrix Factorization](https://arxiv.org/pdf/1906.05108.pdf)

[(Clustering) Federated Clustering via Matrix Factorization Models: From Model Averaging to Gradient Sharing](https://arxiv.org/pdf/2002.04930.pdf)

[Privacy Threats Against Federated Matrix Factorization](https://arxiv.org/pdf/2007.01587.pdf)

### GBDT (Gradient Boosting Decision Trees)

[Practical Federated Gradient Boosting Decision Trees. AAAI 2020.](https://arxiv.org/pdf/1911.04206.pdf)

[Federated Extra-Trees with Privacy Preserving](https://arxiv.org/pdf/2002.07323.pdf)

[SecureGBM: Secure Multi-Party Gradient Boosting](https://arxiv.org/pdf/1911.11997.pdf)

[Federated Forest](https://arxiv.org/pdf/1905.10053.pdf)

[The Tradeoff Between Privacy and Accuracy in Anomaly Detection Using Federated XGBoost](https://arxiv.org/pdf/1907.07157.pdf)

### Other Model
[Privacy Preserving QoE Modeling using Collaborative Learning](https://arxiv.org/pdf/1906.09248.pdf)


[Distributed Dual Coordinate Ascent in General Tree Networks and Its Application in Federated Learning](https://arxiv.org/pdf/1703.04785.pdf)

## Natural language Processing
[Federated pretraining and fine tuning of BERT using clinical notes from multiple silos](https://arxiv.org/pdf/2002.08562.pdf)

[Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/pdf/1811.03604.pdf)

[Federated Learning for Keyword Spotting](https://arxiv.org/pdf/1810.05512.pdf)

[generative sequence models (e.g., language models)](https://arxiv.org/pdf/2006.07490.pdf)

[Pretraining Federated Text Models for Next Word Prediction](https://arxiv.org/pdf/2005.04828.pdf)

[FedNER: Privacy-preserving Medical Named Entity Recognition with Federated Learning. MSRA. 2020-03.](https://arxiv.org/pdf/2003.09288.pdf)

[Federated Learning of N-gram Language Models. Google. ACL 2019.](https://www.aclweb.org/anthology/K19-1012.pdf)

[Federated User Representation Learning](https://arxiv.org/pdf/1909.12535.pdf)

[Two-stage Federated Phenotyping and Patient Representation Learning](https://arxiv.org/pdf/1908.05596.pdf)

[Federated Learning for Emoji Prediction in a Mobile Keyboard](https://arxiv.org/pdf/1906.04329.pdf)

[Federated AI lets a team imagine together: Federated Learning of GANs](https://arxiv.org/pdf/1906.03595.pdf)

[Federated Learning Of Out-Of-Vocabulary Words](https://arxiv.org/pdf/1903.10635.pdf)

[Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/pdf/1812.07108.pdf)

[Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/pdf/1812.02903.pdf)

[Federated Learning for Ranking Browser History Suggestions](https://arxiv.org/pdf/1911.11807.pdf)

## Computer Vision
[Federated Face Anti-spoofing](https://arxiv.org/pdf/2005.14638.pdf)

[(*) Federated Visual Classification with Real-World Data Distribution. MIT. ECCV 2020. 2020-03](https://arxiv.org/pdf/2003.08082.pdf)

[FedVision: An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/pdf/2001.06202.pdf)

## Health Care: 
[Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/pdf/1810.04304.pdf)

[Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data](https://arxiv.org/pdf/1810.08553.pdf)

[Privacy-Preserving Technology to Help Millions of People: Federated Prediction Model for Stroke Prevention](https://arxiv.org/pdf/2006.10517.pdf)

[A Federated Learning Framework for Healthcare IoT devices](https://arxiv.org/pdf/2005.05083.pdf)
Keywords: Split Learning + Sparsification

[Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/pdf/2004.12321.pdf)

[The Future of Digital Health with Federated Learning](https://arxiv.org/pdf/2003.08119.pdf)

[Anonymizing Data for Privacy-Preserving Federated Learning. ECAI 2020.](https://arxiv.org/pdf/2002.09096.pdf)

[Federated machine learning with Anonymous Random Hybridization (FeARH) on medical records](https://arxiv.org/pdf/2001.09751.pdf)

[Stratified cross-validation for unbiased and privacy-preserving federated learning](https://arxiv.org/pdf/2001.08090.pdf)

[Multi-site fMRI Analysis Using Privacy-preserving Federated Learning and Domain Adaptation: ABIDE Results](https://arxiv.org/pdf/2001.05647.pdf)

[Learn Electronic Health Records by Fully Decentralized Federated Learning](https://arxiv.org/pdf/1912.01792.pdf)

[Preserving Patient Privacy while Training a Predictive Model of In-hospital Mortality](https://arxiv.org/pdf/1912.00354.pdf)

[Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270.pdf)

[Federated and Differentially Private Learning for Electronic Health Records](https://arxiv.org/pdf/1911.05861.pdf)

[A blockchain-orchestrated Federated Learning architecture for healthcare consortia](https://arxiv.org/pdf/1910.12603.pdf)

[Federated Uncertainty-Aware Learning for Distributed Hospital EHR Data](https://arxiv.org/pdf/1910.12191.pdf)

[Stochastic Channel-Based Federated Learning for Medical Data Privacy Preserving](https://arxiv.org/pdf/1910.11160.pdf)

[Differential Privacy-enabled Federated Learning for Sensitive Health Data](https://arxiv.org/pdf/1910.02578.pdf)

[LoAdaBoost: Loss-based AdaBoost federated machine learning with reduced computational complexity on IID and non-IID intensive care data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230706)

[Privacy Preserving Stochastic Channel-Based Federated Learning with Neural Network Pruning](https://arxiv.org/pdf/1910.02115.pdf)

[Confederated Machine Learning on Horizontally and Vertically Separated Medical Data for Large-Scale Health System Intelligence](https://arxiv.org/pdf/1910.02109.pdf)

[Privacy-preserving Federated Brain Tumour Segmentation](https://arxiv.org/pdf/1910.00962.pdf)

[HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/pdf/1909.05784.pdf)

[FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/pdf/1907.09173.pdf)

[Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://arxiv.org/pdf/1903.09296.pdf)

[LoAdaBoost:Loss-Based AdaBoost Federated Machine Learning on medical Data](https://arxiv.org/pdf/1811.12629.pdf)

[FADL:Federated-Autonomous Deep Learning for Distributed Electronic Health Record](https://arxiv.org/pdf/1811.11400.pdf)


## Transportation:
[Federated Learning for Vehicular Networks](https://arxiv.org/pdf/2006.01412.pdf)

[Towards Federated Learning in UAV-Enabled Internet of Vehicles: A Multi-Dimensional Contract-Matching Approach](https://arxiv.org/pdf/2004.03877.pdf)

[Federated Learning Meets Contract Theory: Energy-Efficient Framework for Electric Vehicle Networks](https://arxiv.org/pdf/2004.01828.pdf)

[Beyond privacy regulations: an ethical approach to data usage in transportation. TomTom. 2020-04-01](https://arxiv.org/pdf/2004.00491.pdf)

[Privacy-preserving Traffic Flow Prediction: A Federated Learning Approach](https://arxiv.org/pdf/2003.08725.pdf)

[Communication-Efficient Massive UAV Online Path Control: Federated Learning Meets Mean-Field Game Theory. 2020-03](https://arxiv.org/pdf/2003.04451.pdf)

[FedLoc: Federated Learning Framework for Data-Driven Cooperative Localization and Location Data Processing. 2020-03](https://arxiv.org/pdf/2003.03697.pdf)

[Practical Privacy Preserving POI Recommendation](https://arxiv.org/pdf/2003.02834.pdf)

[Federated Learning for Localization: A Privacy-Preserving Crowdsourcing Method](https://arxiv.org/pdf/2001.01911.pdf)

[Federated Transfer Reinforcement Learning for Autonomous Driving](https://arxiv.org/pdf/1910.06001.pdf)

[Energy Demand Prediction with Federated Learning for Electric Vehicle Networks](https://arxiv.org/pdf/1909.00907.pdf)

[Distributed Federated Learning for Ultra-Reliable Low-Latency Vehicular Communications](https://arxiv.org/pdf/1807.08127.pdf)

[Federated Learning for Ultra-Reliable Low-Latency V2V Communications](https://arxiv.org/pdf/1805.09253.pdf)

[Federated Learning in Vehicular Edge Computing: A Selective Model Aggregation Approach](https://ieeexplore.ieee.org/abstract/document/8964354/)


## Recommendation System
[(*) Federated Multi-view Matrix Factorization for Personalized Recommendations](https://arxiv.org/pdf/2004.04256.pdf)


[Robust Federated Recommendation System](https://arxiv.org/pdf/2006.08259.pdf)

[Federated Recommendation System via Differential Privacy](https://arxiv.org/pdf/2005.06670.pdf)

[FedRec: Privacy-Preserving News Recommendation with Federated Learning. MSRA. 2020-03](https://arxiv.org/pdf/2003.09592.pdf)

[Federating Recommendations Using Differentially Private Prototypes](https://arxiv.org/pdf/2003.00602.pdf)

[Meta Matrix Factorization for Federated Rating Predictions](https://arxiv.org/pdf/1910.10086.pdf)

[Federated Hierarchical Hybrid Networks for Clickbait Detection](https://arxiv.org/pdf/1906.00638.pdf)

[Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System](https://arxiv.org/pdf/1901.09888.pdf)

## Speech Recognition
[Training Keyword Spotting Models on Non-IID Data with Federated Learning](https://arxiv.org/pdf/2005.10406.pdf)

## Finance
[FedCoin: A Peer-to-Peer Payment System for Federated Learning](https://arxiv.org/pdf/2002.11711.pdf)

[Towards Federated Graph Learning for Collaborative Financial Crimes Detection](https://arxiv.org/pdf/1909.12946.pdf)

## Smart City
[Cloud-based Federated Boosting for Mobile Crowdsensing](https://arxiv.org/pdf/2005.05304.pdf)

[Exploiting Unlabeled Data in Smart Cities using Federated Learning](https://arxiv.org/pdf/2001.04030.pdf)

## Robotics
[Federated Imitation Learning: A Privacy Considered Imitation Learning Framework for Cloud Robotic Systems with Heterogeneous Sensor Data](https://arxiv.org/pdf/1909.00895.pdf)

[Lifelong Federated Reinforcement Learning: A Learning Architecture for Navigation in Cloud Robotic Systems](https://arxiv.org/pdf/1901.06455.pdf)

## Networking
[A Federated Learning Approach for Mobile Packet Classification](https://arxiv.org/pdf/1907.13113.pdf)

## Blockchain
[Blockchained On-Device Federated Learning](https://arxiv.org/pdf/1808.03949.pdf)

[Record and reward federated learning contributions with blockchain](https://mblocklab.com/RecordandReward.pdf)

## Other
[Boosting Privately: Privacy-Preserving Federated Extreme Boosting for Mobile Crowdsensing](https://arxiv.org/pdf/1907.10218.pdf)

[Self-supervised audio representation learning for mobile devices](https://arxiv.org/pdf/1905.11796.pdf)

[Combining Federated and Active Learning for Communication-efficient Distributed Failure Prediction in Aeronautics](https://arxiv.org/pdf/2001.07504.pdf)

[PMF: A Privacy-preserving Human Mobility Prediction Framework via Federated Learning](https://vonfeng.github.io/files/UbiComp2020_PMF_Final.pdf)

[Federated Multi-task Hierarchical Attention Model for Sensor Analytics](https://arxiv.org/pdf/1905.05142.pdf)

[DÏoT: A Federated Self-learning Anomaly Detection System for IoT](https://arxiv.org/pdf/1804.07474.pdf)

# Benchmark, Dataset and Survey 

## Benchmark and Dataset

[The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems](https://arxiv.org/pdf/2006.07856.pdf)

[Evaluation Framework For Large-scale Federated Learning](https://arxiv.org/pdf/2003.01575.pdf)

[(*) PrivacyFL: A simulator for privacy-preserving and secure federated learning. MIT CSAIL.](https://arxiv.org/pdf/2002.08423.pdf)

[Revocable Federated Learning: A Benchmark of Federated Forest](https://arxiv.org/pdf/1911.03242.pdf)

[Real-World Image Datasets for Federated Learning](https://arxiv.org/pdf/1910.11089.pdf)

[LEAF: A Benchmark for Federated Settings](https://arxiv.org/pdf/1812.01097.pdf)

[Functional Federated Learning in Erlang (ffl-erl)](https://arxiv.org/pdf/1808.08143.pdf)

## Survey

[A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/pdf/1907.09693.pdf)

Researcher: Bingsheng He, NUS [Qinbin Li, PhD, NUS, HKUST](https://qinbinli.com/files/CV_QB.pdf)

[SECure: A Social and Environmental Certificate for AI Systems](https://arxiv.org/pdf/2006.06217.pdf)

[From Federated Learning to Fog Learning: Towards Large-Scale Distributed Machine Learning in Heterogeneous Wireless Networks](https://arxiv.org/pdf/2006.03594.pdf)

[Federated Learning for 6G Communications: Challenges, Methods, and Future Directions](https://arxiv.org/pdf/2006.02931.pdf)

[A Review of Privacy Preserving Federated Learning for Private IoT Analytics](https://arxiv.org/pdf/2004.11794.pdf)

[Survey of Personalization Techniques for Federated Learning. 2020-03-19](https://arxiv.org/pdf/2003.08673.pdf)

[Threats to Federated Learning: A Survey](https://arxiv.org/pdf/2003.02133.pdf)

[Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective](https://arxiv.org/pdf/2002.11545.pdf)

[Federated Learning for Resource-Constrained IoT Devices: Panoramas and State-of-the-art](https://arxiv.org/pdf/2002.10610.pdf)

[Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)

[Privacy-Preserving Blockchain Based Federated Learning with Differential Data Sharing](https://arxiv.org/pdf/1912.04859.pdf)

[An Introduction to Communication Efficient Edge Machine Learning](https://arxiv.org/pdf/1912.01554.pdf)

[Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270.pdf)

[Federated Learning for Coalition Operations](https://arxiv.org/pdf/1910.06799.pdf)

[Federated Learning in Mobile Edge Networks: A Comprehensive Survey](https://arxiv.org/pdf/1909.11875.pdf)

[Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/pdf/1908.07873.pdf)

[A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/pdf/1907.09693.pdf)

[Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)

[No Peek: A Survey of private distributed deep learning](https://arxiv.org/pdf/1812.03288.pdf)

[Communication-Efficient Edge AI: Algorithms and Systems](http://arxiv.org/pdf/2002.09668.pdf)
