import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

import os

class DatasetFromNumpy(Dataset):
    """For MedMNIST"""
    def __init__(self, x, y, transform=None):
        self.transform = transform
        self.all_xs = x
        if len(y.shape) > 1:
            y = np.squeeze(y)
        self.all_ys = y

    def __len__(self):
        return len(self.all_ys)

    def __getitem__(self, idx):
        x = self.all_xs[idx]
        tensor_x = self.transform(x)
        return tensor_x, self.all_ys[idx]

def get_labels(dataset):
    ys = []
    for x, y in dataset:
        ys.append(y)
    return np.array(ys)

def get_stats(dataset, return_labels=False):
    ys = get_labels(dataset)
    unique_labels = np.unique(ys)
    stats = {str(l):0 for l in unique_labels}
    for l in ys:
        stats[str(l)] += 1
    stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[0])}
    if return_labels:
        return stats, ys
    return stats

def get_federated_datasets_dict(args, save_dist_fig=False):
    print("==> preparing %s %s data..." % (args.data, args.distribution))
    np.random.seed(args.seed)
    if args.data == "mnist":
        norm_shape = (0.5,)
        N_classes = 10
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_shape, norm_shape)])
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif args.data == "cifar10":
        N_classes = 10
        norm_shape = (0.5, 0.5, 0.5)
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize(norm_shape, norm_shape)])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        # medmnist (support pathmnist, organmnist_axial)
        data = np.load("./data/%s_train_test.npz" % args.data)
        N_classes = len(np.unique(data["y_train"]))
        norm_shape = (0.5,) if (len(data["x_train"].shape) == 3) or (len(data["x_train"].shape) == 4 and data["x_train"].shape[1] == 1) else (0.5, 0.5, 0.5)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_shape, norm_shape)])
        trainset = DatasetFromNumpy(data["x_train"], data["y_train"], transform=transform)
        testset = DatasetFromNumpy(data["x_test"], data["y_test"], transform=transform)

    assert args.distribution in ["non-iid-practical", "non-iid-pathological"], "Please check data distribution argument!"
    
    def split2clients(dataset, partitions, N_classes):
        stats, ys = get_stats(dataset, return_labels=True)
        splits = [] # N_classes * N_clients
        for i in range(N_classes):
            indices = np.where(ys == i)[0]
            np.random.shuffle(indices)
            cuts = np.cumsum((partitions[i] * stats[str(i)]).astype(int))
            cuts = np.clip(cuts, 0, stats[str(i)])
            splits.append(np.split(indices, cuts))
            
        N_clients = partitions.shape[1]
        clients = []
        for i in range(N_clients):
            indices = np.concatenate([splits[j][i] for j in range(N_classes)], axis=0)
            clients.append(Subset(dataset, indices))
        return clients

    if args.distribution == "non-iid-pathological":
        # 12 clients, each client 2 classes, choose class uniformally
        N_clients = 12
        N_classes_each_client = 2
        N_times = N_clients * N_classes_each_client // N_classes

        cls_list = np.array(list(range(N_classes)))
        order = []
        for i in range(N_times + 1):
            cls_list_copy = cls_list.copy()
            np.random.shuffle(cls_list_copy)
            order.append(cls_list_copy)
        order = np.concatenate(order)[:N_clients * N_classes_each_client]
        clients_classes = np.split(order, [N_classes_each_client * i for i in range(1, N_clients)])
        partitions = np.zeros((N_classes, N_clients))
        for client_idx, c in enumerate(clients_classes):
            for i in range(N_classes_each_client):
                partitions[c[i]][client_idx] = 1
        occurrences = np.sum(partitions, axis=1).astype(int)
        for c in range(N_classes):
            portion = []
            for i in range(occurrences[c]):
                portion.append(np.random.rand())
            indices = np.where(partitions[c] == 1)[0]
            for i, idx in enumerate(indices):
                partitions[c][idx] = portion[i] / np.sum(portion)

    else:
        # non-iid-practical: 1% x 10 + 10% + 80%, also 12 clients
        partition_for_c = np.array([0.01 for _ in range(10)] + [0.1, 0.8])
        partitions = []
        for i in range(N_classes):
            np.random.shuffle(partition_for_c)
            partitions.append(partition_for_c.copy())
        partitions = np.array(partitions)

    N_clients = partitions.shape[1]
    train_clients = split2clients(trainset, partitions, N_classes)
    test_clients = split2clients(testset, partitions, N_classes)

    if save_dist_fig:
        # saving the data distribution for the federated datasets
        ys = [get_labels(c) for c in train_clients]
        fig, ax = plt.subplots()
        ax.hist(ys, bins=[-0.5 + i*0.5 for i in range(27)], stacked=True, density=False, label=['client%d' % i for i in range(1, N_clients+1)])
        ax.legend(loc='best')
        plt.xticks([i+0.25 for i in range(N_classes)],['%d' % i for i in range(N_classes)])
        fn = os.path.join(args.data_dist_dir, "%s_%s_classes.png" % (args.data, args.distribution))
        print("==> classes distribution figured saved at %s" % fn)
        fig.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.clf()

        class_stats = [[] for _ in range(N_classes)]
        for client_idx, y in enumerate(ys):
            for c in y:
                class_stats[c].append(client_idx)
        fig, ax = plt.subplots()
        ax.hist(class_stats, bins=[-0.5 + i*0.5 for i in range(N_clients*3)], stacked=True, density=False, label=['class %d' % i for i in range(N_classes)])
        ax.legend(loc='best')
        plt.xticks([i+0.25 for i in range(N_clients)],['client%d' % i for i in range(1, N_clients+1)])
        plt.xticks(rotation=-60)
        fn = os.path.join(args.data_dist_dir, "%s_%s_clients.png" % (args.data, args.distribution))
        print("==> classes distribution figured saved at %s" % fn)
        fig.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.clf()
    print()

    # train-val split
    val_clients = []
    for i, dset in enumerate(train_clients):
        num_train = int(len(dset) * 0.85)
        num_val = len(dset) - num_train
        train_set, val_set = random_split(dset, [num_train, num_val])
        train_clients[i] = train_set
        val_clients.append(val_set)

    train_dict = {"entire-train": torch.utils.data.ConcatDataset([dset for dset in train_clients])}
    val_dict = {"entire-val": torch.utils.data.ConcatDataset([dset for dset in val_clients])}
    test_dict = {"entire-test": testset}

    train_dict.update({"client%d" % (i+1): train_clients[i] for i in range(N_clients)})
    val_dict.update({"client%d" % (i+1): val_clients[i] for i in range(N_clients)})
    test_dict.update({"client%d" % (i+1): test_clients[i] for i in range(N_clients)})
    return train_dict, val_dict, test_dict