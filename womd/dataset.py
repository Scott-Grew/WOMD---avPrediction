import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from womd import contract

SHARD_SUFFIX = ".npz"


def write_shard(path, samples):
    if len(samples) == 0:
        raise ValueError("refusing to write an empty shard")
    stacked = {
        name: np.stack([np.asarray(sample[name]) for sample in samples])
        for name in contract.SAMPLE_ARRAY_SPEC
    }
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stacked)
    return len(samples)


def shard_paths(directory):
    return sorted(pathlib.Path(directory).glob(f"*{SHARD_SUFFIX}"))


class ShardedSampleDataset(Dataset):
    def __init__(self, directory):
        self.paths = shard_paths(directory)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"no shards under {directory}")
        self.shard_lengths = []
        for path in self.paths:
            with np.load(path) as shard:
                self.shard_lengths.append(len(shard["future_mask"]))
        self.shard_offsets = np.cumsum([0] + self.shard_lengths)
        self._cached_shard_index = None
        self._cached_shard = None

    def __len__(self):
        return int(self.shard_offsets[-1])

    def _shard_for(self, index):
        shard_index = int(np.searchsorted(self.shard_offsets, index, side="right") - 1)
        if shard_index != self._cached_shard_index:
            self._cached_shard = dict(np.load(self.paths[shard_index]))
            self._cached_shard_index = shard_index
        return self._cached_shard, index - int(self.shard_offsets[shard_index])

    def __getitem__(self, index):
        shard, local_index = self._shard_for(index)
        sample = {}
        for name in contract.SAMPLE_ARRAY_SPEC:
            value = shard[name][local_index]
            if value.dtype == np.bool_:
                sample[name] = torch.from_numpy(np.ascontiguousarray(value))
            else:
                sample[name] = torch.from_numpy(np.ascontiguousarray(value.astype(np.float32)))
        return sample


class ShardLocalShuffleSampler(Sampler):
    def __init__(self, sharded_dataset, seed=0):
        self.shard_offsets = sharded_dataset.shard_offsets
        self.shard_lengths = sharded_dataset.shard_lengths
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return int(self.shard_offsets[-1])

    def __iter__(self):
        random_generator = np.random.default_rng(self.seed + self.epoch)
        shard_order = random_generator.permutation(len(self.shard_lengths))
        self.epoch += 1
        for shard_index in shard_order:
            start = int(self.shard_offsets[shard_index])
            length = self.shard_lengths[shard_index]
            for local_index in random_generator.permutation(length):
                yield start + int(local_index)


def samples_to_batch(samples):
    return {
        name: torch.stack([sample[name] for sample in samples])
        for name in contract.SAMPLE_ARRAY_SPEC
    }
