# Data

This directory should contain all the datasets before the federated split, such as MNIST, CIFAR10 (e.g. downloaded from `torchvision`), and the train-test split `.npz` files of the datasets from the [MedMNIST collection](https://arxiv.org/abs/2110.14795).

For the two datasets in the MedMNIST collection, namely the OrganMNIST (axial) dataset and PathMNIST dataset, one can first download them from the [MedMNIST website](https://medmnist.com/), as `./organmnist_axial.npz` and `./pathmnist.npz`. Then, create a train-test split `.npz` file by running `python split_medmnist.py` in this directory.

