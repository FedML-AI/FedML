import collections
import functools
import queue
import threading

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    "FutureResult",
    "SlavePipe",
    "SyncMaster",
    "SynchronizedBatchNorm1d",
    "SynchronizedBatchNorm2d",
    "SynchronizedBatchNorm3d",
    "CallbackContext",
    "execute_replication_callbacks",
    "DataParallelWithCallback",
    "patch_replication_callback",
]


class FutureResult(object):
    """A thread-safe future implementation used for one-to-one communication.

    This class provides a thread-safe mechanism for transferring results between threads,
    typically in a producer-consumer pattern. It is designed for one-to-one communication
    and ensures that the result is safely passed from one thread to another.

    Args:
        None

    Attributes:
        _result: The result value stored in the future.
        _lock: A lock to ensure thread safety.
        _cond: A condition variable associated with the lock for waiting and notifying.

    Methods:
        put(result):
            Puts a result value into the future. If a result already exists, it raises an
            assertion error.
        
        get():
            Retrieves the result value from the future. If the result is not available yet,
            it blocks until the result is put into the future.

    Example:
        Here's an example of using `FutureResult` for communication between two threads:

        ```python
        import threading

        def producer(future):
            result = 42  # Some computation or value to produce
            future.put(result)

        def consumer(future):
            result = future.get()
            print(f"Received result: {result}")

        future = FutureResult()

        # Start the producer and consumer threads
        producer_thread = threading.Thread(target=producer, args=(future,))
        consumer_thread = threading.Thread(target=consumer, args=(future,))

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()
        ```

    Note:
        This class is intended for one-to-one communication between threads.
    """
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        """Put a result into the future.

        Args:
            result: The result value to be stored in the future.

        Raises:
            AssertionError: If a result is already present in the future.
        """
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        """Get the result from the future, blocking if necessary.

        Returns:
            The result value stored in the future.
        """
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple("MasterRegistry", ["result"])
_SlavePipeBase = collections.namedtuple(
    "_SlavePipeBase", ["identifier", "queue", "result"]
)


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication in a multi-threaded environment.

    This class represents a pipe used for communication between a master thread and one
    or more slave threads. It is designed for multi-threaded applications where the
    master thread delegates tasks to the slave threads and waits for their results.

    Args:
        queue (Queue): A queue for sending messages from the slave thread to the master.
        result (FutureResult): A FutureResult object for receiving results from the slave.
        identifier (int): An identifier for the slave thread.

    Attributes:
        queue (Queue): A queue for sending messages from the slave thread to the master.
        result (FutureResult): A FutureResult object for receiving results from the slave.
        identifier (int): An identifier for the slave thread.

    Methods:
        run_slave(msg):
            Executes a task in the slave thread and sends a message to the master thread.
            It waits for the master to acknowledge the completion of the task and returns
            the result.

    Example:
        Here's an example of using `SlavePipe` for master-slave communication:

        ```python
        import threading

        def slave_function(pipe):
            # Perform some computation and send the result to the master
            result = 42  # Placeholder for the result
            pipe.run_slave(result)

        # Create a SlavePipe for communication
        slave_pipe = SlavePipe(queue, result, 1)

        # Start the slave thread
        slave_thread = threading.Thread(target=slave_function, args=(slave_pipe,))
        slave_thread.start()

        # Master thread can send tasks and receive results using the slave_pipe
        task_result = slave_pipe.run_slave(task_data)

        # Wait for the slave thread to finish
        slave_thread.join()

        # Use the task_result received from the slave
        print(f"Received result from slave: {task_result}")
        ```

    Note:
        This class is intended for use in multi-threaded applications where a master
        thread communicates with one or more slave threads.
    """
    def run_slave(self, msg):
        """Execute a task in the slave thread and communicate with the master.

        Args:
            msg: The message or task to be sent to the master.

        Returns:
            The result of the task received from the master.
        """
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object for coordinating communication between master and slave devices.

    In a data parallel setting, the `SyncMaster` object manages the communication between the master device
    and multiple slave devices. It provides a mechanism for slave devices to register and communicate with
    the master during forward and backward passes.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.

    Args:
        master_callback (callable): A callback function to be invoked after collecting messages from slave devices.

    Attributes:
        _master_callback (callable): A callback function to be invoked after collecting messages from slave devices.
        _queue (queue.Queue): A queue for exchanging messages between master and slave devices.
        _registry (collections.OrderedDict): A registry of slave devices and their associated communication pipes.
        _activated (bool): A flag indicating whether the SyncMaster is activated for communication.

    Methods:
        register_slave(identifier):
            Register a slave device and obtain a `SlavePipe` object for communication with the master device.

        run_master(master_msg):
            Main entry for the master device during each forward pass. Collects messages from all devices,
            invokes the master callback to compute a response message, and sends messages back to each device.

        nr_slaves:
            Property that returns the number of registered slave devices.

    Example:
        Here's an example of using `SyncMaster` for coordinating communication in a data parallel setting:

        ```python
        def master_callback(messages):
            # Compute the master message based on received messages
            master_msg = messages[0][1]
            return [(0, master_msg)]  # Send the same message back to the master

        sync_master = SyncMaster(master_callback)

        # Register slave devices and obtain communication pipes
        slave_pipe1 = sync_master.register_slave(1)
        slave_pipe2 = sync_master.register_slave(2)

        # During the forward pass, master device runs run_master to coordinate communication
        master_msg = "Hello from master"
        response_msg = sync_master.run_master(master_msg)

        # Use the response_msg and coordinate further actions

        # Get the number of registered slave devices
        num_slaves = sync_master.nr_slaves
        ```

    Note:
        This class is intended for use in multi-device data parallel applications where a master device
        coordinates communication with multiple slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {"master_callback": self._master_callback}

    def __setstate__(self, state):
        self.__init__(state["master_callback"])

    def register_slave(self, identifier):
        """Register a slave device with the SyncMaster.

        Args:
            identifier (int): An identifier, usually the device ID.

        Returns:
            SlavePipe: A `SlavePipe` object for communicating with the master device.

        Raises:
            AssertionError: If the SyncMaster is already activated and the queue is not empty.

        Notes:
            This method should be called by slave devices to register themselves with the SyncMaster.
            The returned `SlavePipe` object can be used for communication with the master device.

        Example:
            ```python
            sync_master = SyncMaster(master_callback)
            slave_pipe = sync_master.register_slave(1)
            ```

        """
        if self._activated:
            assert self._queue.empty(), "Queue is not clean before next initialization."
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """Run the master device during each forward pass.

        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: The message that the master wants to send to itself.
                This message will be placed as the first message when calling `master_callback`.

        Returns:
            Any: The message to be sent back to the master device.

        Notes:
            This method is the main entry for the master device during each forward pass.
            It collects messages from all devices, invokes the master callback to compute a response message,
            and sends messages back to each device.

        Example:
            ```python
            master_msg = "Hello from master"
            response_msg = sync_master.run_master(master_msg)
            ```

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, "The first result should belongs to the master."

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        """Get the number of registered slave devices.

        Returns:
            int: The number of registered slave devices.

        Example:
            ```python
            num_slaves = sync_master.nr_slaves
            ```

        """
        return len(self._registry)


def _sum_ft(tensor):
    """Sum over the first and last dimensions of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A tensor with the sum of values over the first and last dimensions.

    Example:
        ```python
        input_tensor = torch.tensor([[1, 2], [3, 4]])
        result = _sum_ft(input_tensor)
        # Result: tensor([10])
        ```

    """
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """Add new dimensions at the front and the tail of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A tensor with new dimensions added at the front and the tail.

    Example:
        ```python
        input_tensor = torch.tensor([1, 2, 3])
        result = _unsqueeze_ft(input_tensor)
        # Result: tensor([[[1]], [[2]], [[3]]])
        ```

    """
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple("_ChildMessage", ["sum", "ssum", "sum_size"])
_MasterMessage = collections.namedtuple("_MasterMessage", ["sum", "inv_std"])


class _SynchronizedBatchNorm(_BatchNorm):
    """Synchronized Batch Normalization for parallel computation.

    This class extends PyTorch's BatchNorm2d to support synchronization for data parallelism.
    It uses a master-slave communication pattern to compute batch statistics efficiently.

    Args:
        num_features (int): Number of features in the input tensor.
        eps (float): Small constant added to the denominator for numerical stability. Default: 1e-5
        momentum (float): Momentum factor for the running statistics. Default: 0.1
        affine (bool): If True, apply learned affine transformation. Default: True

    Note:
        This class is typically used in a data parallel setup where multiple GPUs work together.

    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine
        )

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        """Forward pass through the synchronized batch normalization layer.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized and optionally affine-transformed tensor.

        """
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input**2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(
                _ChildMessage(input_sum, input_ssum, sum_size)
            )
        else:
            mean, inv_std = self._slave_pipe.run_slave(
                _ChildMessage(input_sum, input_ssum, sum_size)
            )

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(
                inv_std * self.weight
            ) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Replicate the synchronized batch normalization layer for data parallelism.

        This method is called during data parallel replication to prepare the layer for parallel computation.

        Args:
            ctx: The context object.
            copy_id (int): Identifier for the replica.

        """

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2 : i * 2 + 2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum.

        Args:
            sum_ (torch.Tensor): Sum of values.
            ssum (torch.Tensor): Sum of squared values.
            size (int): Size of the input batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard-deviation.

        """
        assert (
            size > 1
        ), "BatchNorm computes unbiased standard-deviation, which requires size > 1."
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean.data
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    or Instance Norm.

    Note:
        This layer behaves like the built-in PyTorch BatchNorm1d when used on a single GPU or CPU.

    Args:
        num_features (int): Number of features in the input tensor. `batch_size x num_features [x width]`
        eps (float): A small constant added to the denominator for numerical stability. Default: 1e-5
        momentum (float): The momentum factor used for computing running statistics. Default: 0.1
        affine (bool): If True, learnable affine parameters (gamma and beta) are applied. Default: True

    Shape:
        - Input: (N, C) or (N, C, L)
        - Output: (N, C) or (N, C, L) (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)  # 2D input
        >>> output = m(input)
        >>> input_3d = torch.randn(20, 100, 30)  # 3D input
        >>> output_3d = m(input_3d)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute a replication callback `__data_parallel_replicate__` on each module created by original replication.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`.
    Note that, as all modules are isomorphic, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.
    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.

    Args:
        modules (list): List of replicated modules.

    Examples:
        >>> # Replicate a module and execute replication callbacks
        >>> sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        >>> replicated_sync_bn = DataParallelWithCallback(replicate(sync_bn, device_ids=[0, 1]))
        >>> # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, "__data_parallel_replicate__"):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.
    A replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    the original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`.

    Args:
        module (Module): The module to be parallelized.
        device_ids (list): List of device IDs to use for parallelization.

    Examples:
        >>> # Parallelize a module with a replication callback
        >>> sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        >>> replicated_sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        >>> # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Args:
        data_parallel (DataParallel): The existing DataParallel object to be patched.

    Examples:
        >>> sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        >>> sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        >>> patch_replication_callback(sync_bn)
        # This is equivalent to:
        >>> sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        >>> sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])

    Note:
        This function monkey-patches the `DataParallel` object to add the replication callback
        without the need to create a new `DataParallelWithCallback` object.
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate
