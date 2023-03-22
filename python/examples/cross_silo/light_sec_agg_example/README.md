## Introduction
LightSecAgg: a Lightweight and Versatile Design for Secure Aggregation in Federated Learning. MLSys 2022
https://proceedings.mlsys.org/paper/2022/hash/d2ddea18f00665ce8623e36bd4e3c7c5-Abstract.html

### Abstract

Secure model aggregation is a key component of federated learning (FL) that aims at protecting the privacy of each user’s individual model while allowing for their global aggregation. It can be applied to any aggregation-based FL approach for training a global or personalized model. Model aggregation needs to also be resilient against likely user dropouts in FL systems, making its design substantially more complex. State-of-the-art secure aggregation protocols rely on secret sharing of the random-seeds used for mask generations at the users to enable the reconstruction and cancellation of those belonging to the dropped users. The complexity of such approaches, however, grows substantially with the number of dropped users. We propose a new approach, named LightSecAgg, to overcome this bottleneck by changing the design from random-seed reconstruction of the dropped users'' toone-shot aggregate-mask reconstruction of the active users via mask encoding/decoding''. We show that LightSecAgg achieves the same privacy and dropout-resiliency guarantees as the state-of-the-art protocols while significantly reducing the overhead for resiliency against dropped users. We also demonstrate that, unlike existing schemes, LightSecAgg can be applied to secure aggregation in the asynchronous FL setting. Furthermore, we provide a modular system design and optimized on-device parallelization for scalable implementation, by enabling computational overlapping between model training and on-device encoding, as well as improving the speed of concurrent receiving and sending of chunked masks. We evaluate LightSecAgg via extensive experiments for training diverse models (logistic regression, shallow CNNs, MobileNetV3, and EfficientNet-B0) on various datasets (MNIST, FEMNIST, CIFAR-10, GLD-23K) in a realistic FL system with large number of users and demonstrate that LightSecAgg significantly reduces the total training time.
## Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

At the server side, run the following script:
```
bash run_server.sh your_run_id
```

For client 1, run the following script:
```
bash run_client.sh 1 your_run_id
```
For client 2, run the following script:
```
bash run_client.sh 2 your_run_id
```

## A Better User-experience with FedML MLOps (open.fedml.ai)
To reduce the difficulty and complexity of these CLI commands. We recommend you to use our MLOps (open.fedml.ai).
FedML MLOps provides:
- Install Client Agent and Login
- Inviting Collaborators and group management
- Project Management
- Experiment Tracking (visualizing training results)
- monitoring device status
- visualizing system performance (including profiling flow chart)
- distributed logging
- model serving