import torch
from torch.utils.data import Dataset, Subset, DataLoader
from collections import Counter

import random
import numpy as np
import os

from .mnist.loader import get_all_targets_mnist, get_dataloader_mnist
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from .cifar100.loader import get_all_targets_cifar100, get_dataloader_cifar100
from .cinic10.loader import get_all_targets_cinic10, get_dataloader_cinic10
from .tinyimagenet.loader import (
    get_all_targets_tinyimagenet,
    get_dataloader_tinyimagenet,
)

__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "mnist": get_all_targets_mnist,
    "cifar10": get_all_targets_cifar10,
    "cifar100": get_all_targets_cifar100,
    "cinic10": get_all_targets_cinic10,
    "tinyimagenet": get_all_targets_tinyimagenet,
}
DATA_LOADERS = {
    "mnist": get_dataloader_mnist,
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
    "cinic10": get_dataloader_cinic10,
    "tinyimagenet": get_dataloader_tinyimagenet,
}


class _UnlabeledWrapper(Dataset):
    """
    Wraps a labeled dataset but returns (x, -1). We ignore labels in the unlabeled pool.
    """
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, _ = self.base[self.indices[i]]
        return x, -1  # dummy label


def make_shared_unlabeled_pool(train_dataset, unlabeled_pool_size, *, seed=0):
    """
    Splits `train_dataset` into:
      - Do: shared unlabeled pool (size = unlabeled_pool_size)
      - L : remaining labeled data for clients
    Notes:
      - We randomly sample without replacement, deterministically (seed).
      - For MNIST/Fashion, we keep the SAME transforms as training, so both
        clients and server see consistent pre-processing.
    """
    N = len(train_dataset)
    if unlabeled_pool_size <= 0:
        # No unlabeled pool; use whole dataset for labeled client split
        Do = _UnlabeledWrapper(train_dataset, [])
        L = Subset(train_dataset, list(range(N)))
        return Do, L

    if unlabeled_pool_size >= N:
        raise ValueError(f"unlabeled_pool_size={unlabeled_pool_size} must be < train size {N}")

    rng = np.random.RandomState(seed)
    all_idx = np.arange(N)
    rng.shuffle(all_idx)

    or_idx = all_idx[:unlabeled_pool_size]
    keep_idx = all_idx[unlabeled_pool_size:]

    Do = _UnlabeledWrapper(train_dataset, or_idx)
    L = Subset(train_dataset, keep_idx)
    return Do, L


def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    oracle_size=0,
    oracle_batch_size=None,
    unlabeled_pool_size=0,
    num_workers=0,
    **_,
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """

    root = os.path.join(root, dataset_name)
    all_targets = np.asarray(DATA_INSTANCES[dataset_name](root))
    num_classes = len(np.unique(all_targets))
    net_dataidx_map_test = None

    # --- Shared unlabeled pool (Do) selection FIRST ---
    N = len(all_targets)
    unlabeled_pool_size = int(unlabeled_pool_size or 0)

    all_indices = np.arange(N, dtype="int64")
    Do_indices = np.array([], dtype="int64")

    if unlabeled_pool_size > 0:
        if unlabeled_pool_size >= N:
            raise ValueError(f"unlabeled_pool_size={unlabeled_pool_size} must be < train size {N}")
        np.random.shuffle(all_indices)
        Do_indices = np.asarray(all_indices[:unlabeled_pool_size], dtype="int64")

    # Remaining labeled indices to distribute to clients
    L_indices = np.setdiff1d(all_indices, Do_indices, assume_unique=False)

    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None} for i in range(n_clients)
    }

    if partition.method == "centralized":
        net_dataidx_map = {0: L_indices.copy()}

    elif partition.method == "iid":
        net_dataidx_map = iid_partition_from_indices(L_indices, n_clients)

    elif partition.method == "lda":
        net_dataidx_map = lda_partition_from_indices(
            all_targets=all_targets,
            candidate_indices=L_indices,
            n_clients=n_clients,
            alpha=partition.alpha,
        )

    elif partition.method == "sharding":
        # kept as-is for minimal changes
        net_dataidx_map, rand_set_all = sharding_partition(
            all_targets, n_clients, partition.shard_per_user
        )
        all_targets_test = DATA_INSTANCES[dataset_name](root, train=False)
        net_dataidx_map_test, _ = sharding_partition(
            all_targets_test,
            n_clients,
            partition.shard_per_user,
            rand_set_all=rand_set_all,
        )

    elif partition.method == "sharding_max":
        # kept as-is for minimal changes
        net_dataidx_map = sharding_max_partition(all_targets, n_clients, partition.K)

    else:
        raise NotImplementedError

    print(">>> Distributing client test data...")

    # For minimal changes, only apply post-hoc removal for partition modes
    # that still split the full dataset (sharding / sharding_max).
    if partition.method in ["sharding", "sharding_max"] and Do_indices.size > 0:
        for k, idxs in net_dataidx_map.items():
            idxs = np.asarray(idxs, dtype="int64")
            net_dataidx_map[k] = np.setdiff1d(idxs, Do_indices, assume_unique=False)

    if net_dataidx_map_test is not None:
        for client_idx, dataidxs in net_dataidx_map_test.items():
            local_testloader = DATA_LOADERS[dataset_name](
                root, train=False, batch_size=batch_size, dataidxs=dataidxs,
            )
            local_loaders[client_idx]["test"] = local_testloader
            local_loaders[client_idx]["dist"] = get_dist_vec(
                local_testloader, num_classes
            )

    print(">>> Distributing client train data...")
    for client_idx, dataidxs in net_dataidx_map.items():
        local_loaders[client_idx]["datasize"] = len(dataidxs)
        local_loaders[client_idx]["train"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=batch_size, dataidxs=dataidxs,
        )

    global_loaders = {
        "train": DATA_LOADERS[dataset_name](root, train=True, batch_size=batch_size, num_workers=num_workers),
        "test": DATA_LOADERS[dataset_name](root, train=False, batch_size=batch_size, num_workers=num_workers),
    }

    # Build the unlabeled dataset Do with the exact same train transforms
    train_full_dataset = global_loaders["train"].dataset
    unlabeled_dataset = _UnlabeledWrapper(train_full_dataset, Do_indices.tolist())

    # Count class samples in Clients
    data_map = net_dataidx_map_counter(net_dataidx_map, all_targets)

    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "data_map": data_map,
        "num_classes": num_classes,
        "unlabeled": unlabeled_dataset,
        "unlabeled_indices": Do_indices.tolist(),
    }

    # Set oracle loader for CL-like memory
    oracle_idxs = oracle_partition(all_targets, oracle_size=oracle_size)
    obs = batch_size

    if oracle_batch_size is not None:
        obs = oracle_batch_size

    if oracle_idxs is not None:
        data_distributed["oracle"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=obs, dataidxs=oracle_idxs
        )

    return data_distributed


def centralized_partition(all_targets):
    labels = all_targets
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    tot_idx = np.array(tot_idx)
    np.random.shuffle(tot_idx)
    net_dataidx_map[0] = tot_idx

    return net_dataidx_map


def iid_partition(all_targets, n_clients):
    labels = all_targets
    length = int(len(labels) / n_clients)
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    for client_idx in range(n_clients):
        np.random.shuffle(tot_idx)
        data_idxs = tot_idx[:length]
        tot_idx = tot_idx[length:]
        net_dataidx_map[client_idx] = np.array(data_idxs)

    return net_dataidx_map


def iid_partition_from_indices(candidate_indices, n_clients):
    candidate_indices = np.asarray(candidate_indices, dtype="int64").copy()
    np.random.shuffle(candidate_indices)

    splits = np.array_split(candidate_indices, n_clients)

    net_dataidx_map = {}
    for client_idx in range(n_clients):
        net_dataidx_map[client_idx] = np.asarray(splits[client_idx], dtype="int64")

    return net_dataidx_map


def lda_partition_from_indices(all_targets, candidate_indices, n_clients, alpha, min_require_size=10):
    candidate_indices = np.asarray(candidate_indices, dtype="int64")
    all_targets = np.asarray(all_targets)
    unique_classes = np.unique(all_targets[candidate_indices])

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        N = len(candidate_indices)

        for k in unique_classes:
            idx_k = candidate_indices[all_targets[candidate_indices] == k]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)]
            )

            if proportions.sum() == 0:
                proportions = np.repeat(1.0 / n_clients, n_clients)
            else:
                proportions = proportions / proportions.sum()

            cut_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_chunks = np.split(idx_k, cut_points)

            idx_batch = [
                idx_j + chunk.tolist()
                for idx_j, chunk in zip(idx_batch, split_chunks)
            ]

        min_size = min(len(idx_j) for idx_j in idx_batch)

    net_dataidx_map = {}
    for i in range(n_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = np.asarray(idx_batch[i], dtype="int64")

    return net_dataidx_map


def sharding_partition(all_targets, n_clients, shard_per_user, rand_set_all=[]):
    net_dataidx_map = {i: np.array([], dtype="int64") for i in range(n_clients)}
    idxs_dict = {}

    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

        num_classes = len(np.unique(all_targets))
        shard_per_class = int(shard_per_user * n_clients / num_classes)

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((n_clients, -1))

    # divide and assign
    for i in range(n_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        net_dataidx_map[i] = np.concatenate(rand_set).astype("int")

    test = []
    for key, value in net_dataidx_map.items():
        x = np.unique(torch.tensor(all_targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(all_targets)
    assert len(set(list(test))) == len(all_targets)

    return net_dataidx_map, rand_set_all


def sharding_max_partition(all_targets, n_clients, K):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    shard_size = int(length / K)
    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tmp = []
        while 1:
            tmp.append(idx_by_label[:shard_size])
            idx_by_label = idx_by_label[shard_size:]
            if len(idx_by_label) < shard_size / 2:
                break
        tot_idx_by_label.append(tmp)

    for client_idx in range(n_clients):
        idx_by_devices = []

        while len(idx_by_devices) < K:
            chosen_label = np.random.choice(unique_classes, 1, replace=False)[0]

            if len(tot_idx_by_label[chosen_label]) > 0:
                l_idx = np.random.choice(
                    len(tot_idx_by_label[chosen_label]), 1, replace=False
                )[0]
                idx_by_devices.append(
                    tot_idx_by_label[chosen_label][l_idx].tolist()
                )
                del tot_idx_by_label[chosen_label][l_idx]

        data_idxs = np.concatenate(idx_by_devices)
        np.random.shuffle(data_idxs)
        net_dataidx_map[client_idx] = data_idxs

    return net_dataidx_map


def lda_partition(all_targets, n_clients, alpha):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tot_idx_by_label.append(idx_by_label)

    min_size = 0

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        N, K = len(all_targets), len(np.unique(all_targets))

        for k in range(K):
            idx_k = np.where(all_targets == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(
                N, alpha, n_clients, idx_batch, idx_k
            )

    for i in range(n_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def oracle_partition(all_targets, oracle_size=0):
    oracle_idxs = None

    if oracle_size != 0:
        idxs_dict = {}

        for i in range(len(all_targets)):
            label = torch.tensor(all_targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

            oracle_idxs = []

        for value in idxs_dict.values():
            oracle_idxs += value[0:oracle_size]

    return oracle_idxs


def get_dist_vec(dataloader, num_classes):
    """Calculate distribution vector for local set"""
    targets = dataloader.dataset.targets
    dist_vec = torch.zeros(num_classes)
    counter = Counter(targets)

    for class_idx, count in counter.items():
        dist_vec[class_idx] = count

    dist_vec /= len(targets)

    return dist_vec


def net_dataidx_map_counter(net_dataidx_map, all_targets):
    data_map = [[] for _ in range(len(net_dataidx_map.keys()))]
    num_classes = len(np.unique(all_targets))

    prev_key = -1
    for key, item in net_dataidx_map.items():
        client_class_count = [0 for _ in range(num_classes)]
        class_elems = all_targets[item]
        for elem in class_elems:
            client_class_count[elem] += 1

        data_map[key] = client_class_count

    return np.array(data_map)