import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.columns_ = list(X.columns)
        self.categories_ = []

        for col in self.columns_:
            cats = np.unique(X[col].values)
            cats = np.sort(cats)
            self.categories_.append(cats)

        lengths = [len(c) for c in self.categories_]
        self.feature_indices_ = np.zeros(len(lengths) + 1, dtype=np.int64)
        self.feature_indices_[1:] = np.cumsum(lengths)

        self.value_to_index_ = []
        for cats in self.categories_:
            mapping = {}
            for i, v in enumerate(cats):
                mapping[v] = i
            self.value_to_index_.append(mapping)

        return self

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_objects = X.shape[0]
        total_dimention = int(self.feature_indices_[-1])
        result = np.zeros((n_objects, total_dimention), dtype=self.dtype)

        for j, col in enumerate(self.columns_):
            start = int(self.feature_indices_[j])
            mapping = self.value_to_index_[j]
            col_values = X[col].values

            for i in range(n_objects):
                v = col_values[i]
                idx = mapping.get(v, None)
                if idx is not None:
                    result[i, start + idx] = 1.0

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.columns_ = list(X.columns)
        y = np.asarray(Y)
        self.n_total_ = float(len(y))

        self.sum_y_ = []
        self.count_ = []

        for col in self.columns_:
            col_vals = X[col].values
            sum_y = {}
            cnt = {}
            for v, t in zip(col_vals, y):
                cnt[v] = cnt.get(v, 0.0) + 1.0
                sum_y[v] = sum_y.get(v, 0.0) + float(t)
            self.sum_y_.append(sum_y)
            self.count_.append(cnt)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_objects = X.shape[0]
        n_features = len(self.columns_)
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for j, col in enumerate(self.columns_):
            sum_y = self.sum_y_[j]
            cnt = self.count_[j]
            col_vals = X[col].values

            for i in range(n_objects):
                v = col_vals[i]
                c = float(cnt.get(v, 0.0))
                s = float(sum_y.get(v, 0.0))

                if c > 0.0:
                    successes = s / c
                    counters = c / self.n_total_
                else:
                    successes = 0.0
                    counters = 0.0

                # Исправление деления на ноль
                denominator = counters + b
                if abs(denominator) < 1e-12:  # Проверка на близкое к нулю значение
                    relation = 0.0
                else:
                    relation = (successes + a) / denominator

                base = 3 * j
                result[i, base] = successes
                result[i, base + 1] = counters
                result[i, base + 2] = relation

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.columns_ = list(X.columns)
        n_objects = X.shape[0]

        self.folds_ = list(group_k_fold(n_objects, n_splits=self.n_folds, seed=seed))

        self.encoders_ = []
        for val_idx, train_idx in self.folds_:
            enc = SimpleCounterEncoder(dtype=self.dtype)
            enc.fit(X.iloc[train_idx], Y.iloc[train_idx])
            self.encoders_.append(enc)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_objects = X.shape[0]
        n_features = len(self.columns_)
        result = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for (k, (val_idx, train_idx)) in enumerate(self.folds_):
            enc = self.encoders_[k]
            part = enc.transform(X.iloc[val_idx], a, b)
            result[val_idx, :] = part

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    unique_values = np.unique(x)
    w = np.zeros(len(unique_values), dtype=np.float64)

    for i, v in enumerate(unique_values):
        mask = (x == v)
        if np.any(mask):
            w[i] = y[mask].mean()
        else:
            w[i] = 0.0

    return w
