import os
import shutil

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


class AverageMeter(object):
    """
    Computes and stores the average and sum of values over time.

    Attributes:
        avg (float): The current average value.
        sum (float): The current sum of values.
        cnt (int): The current count of values.

    Methods:
        reset(): Reset the average, sum, and count to zero.
        update(val, n=1): Update the meter with a new value and count.

    """

    def __init__(self):
        """
        Initializes an AverageMeter object with initial values of zero.
        """
        self.reset()

    def reset(self):
        """
        Reset the average, sum, and count to zero.
        """
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value and count.

        Args:
            val (float): The new value to update the meter with.
            n (int): The count associated with the new value. Default is 1.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy of model predictions given the output and target labels.

    Args:
        output (Tensor): The model's output predictions.
        target (Tensor): The ground truth labels.
        topk (tuple of int): The top-k accuracy values to compute. Default is (1,).

    Returns:
        list of float: A list of top-k accuracy values.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    """
    Apply cutout augmentation to an image.

    Args:
        length (int): The size of the cutout square region.

    """

    def __init__(self, length):
        """
        Initializes the Cutout object with a specified cutout length.

        Args:
            length (int): The size of the cutout square region.

        """
        self.length = length

    def __call__(self, img):
        """
        Apply cutout augmentation to an image.

        Args:
            img (PIL.Image): The input image.

        Returns:
            PIL.Image: The augmented image with cutout applied.

        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """
    Define data transformations for CIFAR-10 dataset.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple: A tuple of train and validation data transforms.

    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    """
    Count the number of parameters in a model in megabytes (MB).

    Args:
        model (nn.Module): The model for which to count parameters.

    Returns:
        float: The number of parameters in megabytes (MB).

    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    """
    Save a checkpoint of the model's state.

    Args:
        state (dict): The model's state dictionary.
        is_best (bool): True if this is the best checkpoint, False otherwise.
        save (str): The directory where the checkpoint will be saved.

    """
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    """
    Save the model's state dictionary to a file.

    Args:
        model (nn.Module): The PyTorch model to be saved.
        model_path (str): The path to the file where the model state will be saved.

    """
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    """
    Load a model's state dictionary from a file into the model.

    Args:
        model (nn.Module): The PyTorch model to which the state will be loaded.
        model_path (str): The path to the file containing the model state.

    """
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    """
    Apply dropout to a tensor.

    Args:
        x (Tensor): The input tensor to which dropout will be applied.
        drop_prob (float): The probability of dropping out a value.

    Returns:
        Tensor: The tensor after dropout has been applied.

    """
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    """
    Create an experiment directory and optionally save scripts.

    Args:
        path (str): The directory path for the experiment.
        scripts_to_save (list of str, optional): List of script file paths to save in the directory.

    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)