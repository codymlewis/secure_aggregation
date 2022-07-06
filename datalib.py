import numpy as np


def lda(labels, nclients, nclasses, rng, alpha=0.5):
    r"""
    Latent Dirichlet allocation defined in `https://arxiv.org/abs/1909.06335 <https://arxiv.org/abs/1909.06335>`_
    default value from `https://arxiv.org/abs/2002.06440 <https://arxiv.org/abs/2002.06440>`_
    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nclients)]
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


class DataIter:
    """Iterator that gives random batchs in pairs of $(X_i, y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, X, Y, batch_size, classes, rng):
        """
        Construct a data iterator.
        
        Arguments:
        - X: the samples
        - y: the labels
        - batch_size: the batch size
        - classes: the number of classes
        - rng: the random number generator
        """
        self.X, self.Y = X, Y
        self.batch_size = len(Y) if batch_size is None else min(batch_size, len(Y))
        self.len = len(Y)
        self.classes = classes
        self.rng = rng

    def __iter__(self):
        """Return this as an iterator."""
        return self

    def __next__(self):
        """Get a random batch."""
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.ds)


class Dataset:
    """Object that contains the full dataset, primarily to prevent the need for reloading for each client."""

    def __init__(self, ds):
        """
        Construct the dataset.
        Arguments:
        - ds: a hugging face dataset
        """
        self.ds = ds
        self.classes = len(np.union1d(np.unique(ds['train']['Y']), np.unique(ds['test']['Y'])))
        self.input_shape = ds['train'][0]['X'].shape

    def get_iter(self, split, batch_size=None, idx=None, rng=np.random.default_rng()) -> DataIter:
        """
        Generate an iterator out of the dataset.
        
        Arguments:
        - split: the split to use, either "train" or "test"
        - batch_size: the batch size
        - idx: the indices to use
        - rng: the random number generator
        """
        X, Y = self.ds[split]['X'], self.ds[split]['Y']
        if idx is not None:
            X, Y = X[idx], Y[idx]
        return DataIter(X, Y, batch_size, self.classes, rng)

    def fed_split(self, batch_sizes, mapping=None, rng=np.random.default_rng()):
        """
        Divide the dataset for federated learning.
        
        Arguments:
        - batch_sizes: the batch sizes for each client
        - mapping: a function that takes the dataset information and returns the indices for each client
        - rng: the random number generator
        """
        if mapping is not None:
            distribution = mapping(self.ds['train']['Y'], len(batch_sizes), self.classes, rng)
            return [
                self.get_iter("train", b, idx=d, rng=rng)
                for b, d in zip(batch_sizes, distribution)
            ]
        return [self.get_iter("train", b, rng=rng) for b in batch_sizes]

