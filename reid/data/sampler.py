import copy
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample P identities, then K instances per identity, therefore batch size is P*K.
    Real images and synthetic views are drawn alike.

    Args:
    - data_source (list): list of (path, domain, pid, camid).
    - mini_batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch (K).
    """

    def __init__(self, data_source, mini_batch_size, num_instances, seed=42):
        self.num_instances = num_instances
        self.num_pids_per_batch = mini_batch_size // num_instances
        self.batch_size = mini_batch_size
        self._seed = int(seed)

        self.pid_index = defaultdict(list)
        for index, (_, _, pid, _) in enumerate(data_source):
            self.pid_index[pid].append(index)
        self.pids = sorted(self.pid_index)

        self.length = 0
        for pid in self.pids:
            num = max(len(self.pid_index[pid]), self.num_instances)
            self.length += num - num % self.num_instances

    def __iter__(self):
        np.random.seed(self._seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}
            batch_indices = []
            while len(avl_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
                for pid in selected_pids:
                    if pid not in batch_idxs_dict:
                        batch_idxs_dict[pid] = self._sample_identity(pid)

                    avl_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop(0))

                    if len(avl_idxs) < self.num_instances:
                        avl_pids.remove(pid)

                if len(batch_indices) == self.batch_size:
                    yield from batch_indices
                    batch_indices = []

    def _sample_identity(self, pid):
        idxs = copy.deepcopy(self.pid_index[pid])
        if len(idxs) < self.num_instances:
            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
        np.random.shuffle(idxs)
        return idxs

    def __len__(self):
        return self.length
