# FedML APIs (core) 
FedML-core separates the communication and the model training into two core components. 
The first is the communication protocol component (labeled as distributed in the figure). 
It is responsible for low-level communication among different works in the network. The communication backend is based on MPI (message passing interface, https://pypi.org/project/mpi4py/). 
We consider adding more backends as necessary, such as RPC (remote procedure call). 
Inside the communication protocol component, a TopologyManager supports the flexible topology configuration required by different distributed learning algorithms. 
The second is the on-device deep learning component, which is built based on the popular deep learning framework PyTorch or TensorFlow. 
For flexibility, there is no restriction on the framework for this part. 
Users can implement trainers and coordinators according to their needs. 
In addition, low-level APIs support security and privacy-related algorithms.

The philosophy of the FedML programming interface is to provide the simplest user experience: 
allowing users to build distributed training applications; 
to design customized message flow and topology definitions 
by only focusing on algorithmic implementations while ignoring the low-level communication backend details.

## Worker-Oriented Programming
FedML-core provides the worker-oriented programming design pattern, which can be used to program the worker behavior 
when participating in training or coordination in the FL algorithm. 
We describe it as worker-oriented because its counterpart, the standard distributed training library 
(e.g., [torch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)), 
normally completes distributed training programming by describing the entire training procedure rather than focusing on the behavior of each worker. 

With the worker-oriented programming design pattern, the user can customize its own worker in FL network 
by inheriting the WorkerManager class and utilizing its predefined APIs register_message_receive_handler and send_message to 
define the receiving and sending messages without considering the underlying communication mechanism (as shown in the highlight blue box
in Figure 1). 
Conversely, existing distributed training frameworks do not have such flexibility for algorithm innovation. 
In order to make the comparison clearer, we use the most popular machine learning framework PyTorch as an example. 
Figure 1 illustrates a complete training procedure (distributed synchronous SGD) 
and aggregates gradients from all other workers with the all_reduce messaging passing interface. 
Although it supports multiprocessing training, it cannot flexibly customize different messaging flows in any network topology. 
In PyTorch, another distributed training API, [torch.nn.parallel.paraDistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) also has such inflexibly. 
Note that [torch.distributed.rpc](https://pytorch.org/tutorials/intermediate/rpc\_tutorial.html) is a low-level communication back API 
that can finish any communication theoretically, but it is not user-friendly for federated learning researchers. 

## Message Definition Beyond Gradient and Model
FedML also considers supporting message exchange beyond the gradient or model from the perspective of message flow. 
This type of auxiliary information may be due to either the need for algorithm design or the need for system-wide configuration delivery. 
Each worker defines the message type from the perspective of sending. 
Thus, in the above introduced worker-oriented programming, WorkerManager should handle messages defined by other trainers and also send messages defined by itself. 
The sending message is normally executed after handling the received message. As shown in Figure 2, in the yellow background highlighted code snippet, 
workers can send any message type and related message parameters during the train() function.

## Topology management
?> Code Path: https://github.com/FedML-AI/FedML/tree/master/fedml_core/distributed/topology

As demonstrated in Figure 3, FL has various topology definitions, 
such as vertical FL, split learning, decentralized FL, and Hierarchical FL. 
In order to meet such diverse requirements, FedML provides TopologyManager to manage the topology 
and allow users to send messages to arbitrary neighbors during training. 
Specifically, after the initial setting of TopologyManager is completed, 
for each trainer in the network, the neighborhood worker ID can be queried through the TopologyManager. 
In line 26 of Figure 2, we see that the trainer can query its neighbor nodes 
through the TopologyManager before sending its message. 


?>  `BaseTopologyManager`  is the abstract class that defines five interfaces as follows.

``` python
import abc


class BaseTopologyManager(abc.ABC):

    @abc.abstractmethod
    def generate_topology(self):
        pass

    @abc.abstractmethod
    def get_in_neighbor_idx_list(self, node_index):
        pass

    @abc.abstractmethod
    def get_out_neighbor_idx_list(self, node_index):
        pass

    @abc.abstractmethod
    def get_in_neighbor_weights(self, node_index):
        pass

    @abc.abstractmethod
    def get_out_neighbor_weights(self, node_index):
        pass

```


?>  In out current version, the predefined classes that inherit from `BaseTopologyManager` 
include `SymmetricTopologyManager` and `AsymmetricTopologyManager`. `SymmetricTopologyManager` supports symmetric topology 
which has the same IN and OUT neighbor list, while `AsymmetricTopologyManager` has asymmetric topology that IN and OUT neighbor list are different.   
`SymmetricTopologyManager` and `AsymmetricTopologyManager` also differs in the initialization method.

``` python
class SymmetricTopologyManager(BaseTopologyManager):
    """
    The topology definition is determined by this initialization method.

    Arguments:
        n (int): number of nodes in the topology.
        neighbor_num (int): number of neighbor for each node
    """
    def __init__(self, n, neighbor_num=2):
        ...
```
``` python
class AsymmetricTopologyManager(BaseTopologyManager):

    """
    The topology definition is determined by this initialization method.

    Arguments:
        n (int): number of nodes in the topology.
        undirected_neighbor_num (int): number of undirected (symmetric) neighbors for each node
        out_directed_neighbor (int): number of out (asymmetric) neighbors for each node
    """
    def __init__(self, n, undirected_neighbor_num=3, out_directed_neighbor=3):
        ...
```

?> The examples below show the API calling that constructs a ring topology with `SymmetricTopologyManager`, and a asymmetric topology with `AsymmetricTopologyManager`.
``` python
    # generate a ring topology
    tpmgr = SymmetricTopologyManager(6, 2)
    tpmgr.generate_topology()
    print("tpmgr.topology = " + str(tpmgr.topology))

    # get the OUT neighbor weights for node 1
    out_neighbor_weights = tpmgr.get_out_neighbor_weights(1)
    print("out_neighbor_weights = " + str(out_neighbor_weights))

    # get the OUT neighbor index list for node 1
    out_neighbor_idx_list = tpmgr.get_out_neighbor_idx_list(1)
    print("out_neighbor_idx_list = " + str(out_neighbor_idx_list))

    # get the IN neighbor weights for node 1
    in_neighbor_weights = tpmgr.get_in_neighbor_weights(1)
    print("in_neighbor_weights = " + str(in_neighbor_weights))

    # get the IN neighbor index list for node 1
    in_neighbor_idx_list = tpmgr.get_in_neighbor_idx_list(1)
    print("in_neighbor_idx_list = " + str(in_neighbor_idx_list))

    # The result is: 
    tpmgr.topology = [[0.33333334 0.33333334 0.         0.         0.         0.33333334]
     [0.33333334 0.33333334 0.33333334 0.         0.         0.        ]
     [0.         0.33333334 0.33333334 0.33333334 0.         0.        ]
     [0.         0.         0.33333334 0.33333334 0.33333334 0.        ]
     [0.         0.         0.         0.33333334 0.33333334 0.33333334]
     [0.33333334 0.         0.         0.         0.33333334 0.33333334]]
    out_neighbor_weights = [0.33333334 0.33333334 0.33333334 0.         0.         0.        ]
    out_neighbor_idx_list = [0, 2]
    in_neighbor_weights = [0.33333334 0.33333334 0.33333334 0.         0.         0.        ]
    in_neighbor_idx_list = [0, 2]
```
``` python
    # generate a asymmetric topology
    tpmgr = AsymmetricTopologyManager(8, 4, 2)
    tpmgr.generate_topology()
    print(tpmgr.topology)

    # get the OUT neighbor weights for node 1
    out_neighbor_weights = tpmgr.get_out_neighbor_weights(1)
    print(out_neighbor_weights)

    # get the OUT neighbor index list for node 1
    out_neighbor_idx_list = tpmgr.get_out_neighbor_idx_list(1)
    print(out_neighbor_idx_list)

    # get the IN neighbor weights for node 1
    in_neighbor_weights = tpmgr.get_in_neighbor_weights(1)
    print(in_neighbor_weights)

    # get the IN neighbor index list for node 1
    in_neighbor_idx_list = tpmgr.get_in_neighbor_idx_list(1)
    print(in_neighbor_idx_list)

    # the result is:
    tpmgr.topology = [[0.16666667 0.16666667 0.16666667 0.         0.16666667 0.
      0.16666667 0.16666667]
     [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.
      0.         0.16666667]
     [0.14285715 0.14285715 0.14285715 0.14285715 0.14285715 0.14285715
      0.         0.14285715]
     [0.14285715 0.14285715 0.14285715 0.14285715 0.14285715 0.14285715
      0.14285715 0.        ]
     [0.         0.         0.2        0.2        0.2        0.2
      0.2        0.        ]
     [0.         0.16666667 0.         0.16666667 0.16666667 0.16666667
      0.16666667 0.16666667]
     [0.16666667 0.         0.16666667 0.         0.16666667 0.16666667
      0.16666667 0.16666667]
     [0.14285715 0.14285715 0.         0.14285715 0.14285715 0.14285715
      0.14285715 0.14285715]]
    out_neighbor_weights = [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.
     0.         0.16666667]
    out_neighbor_idx_list = [0, 2, 3, 4, 7]
    in_neighbor_weights = [0.16666667, 0.16666667, 0.14285715, 0.14285715, 0.0, 0.16666667, 0.0, 0.14285715]
    in_neighbor_idx_list = [0, 2, 3, 5, 7]
    
```

## Trainer and coordinator
We also need the coordinator to complete the training (\textit{e.g.}, in the FedAvg algorithm, the central worker is the coordinator while the others are trainers). For the trainer and coordinator, \texttt{FedML} does not over-design. Rather, it gives the implementation completely to the developers, reflecting the flexibility of our framework. The implementation of the trainer and coordinator is similar to the process in Figure \ref{fig:overview_training_oriented}, which is completely consistent with the training implementation of a standalone version training. We provide some reference implementations of different trainers and coordinators in our source code (Section \ref{sec:reference_examples}).
