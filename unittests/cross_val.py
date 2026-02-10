import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, the 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    idx_array = np.arange(num_objects)
    fold_size = num_objects // num_folds
    split_indices = []

    for fold_idx in range(num_folds):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < num_folds - 1 else num_objects

        val_indices = idx_array[val_start:val_end]
        train_indices = np.concatenate([idx_array[:val_start], idx_array[val_end:]])

        split_indices.append((train_indices, val_indices))

    return split_indices


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    results = {}

    # Iterate through all parameter combinations
    for normalizer, norm_name in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:

                    scores = []

                    # Cross-validation
                    for train_idx, val_idx in folds:
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        # Apply normalization if provided
                        if normalizer is not None:
                            norm = type(normalizer)()
                            norm.fit(X_train)
                            X_train_norm = norm.transform(X_train)
                            X_val_norm = norm.transform(X_val)
                        else:
                            X_train_norm = X_train
                            X_val_norm = X_val

                        # Create and train model
                        model = knn_class(
                            n_neighbors=n_neighbors,
                            metric=metric,
                            weights=weight
                        )
                        model.fit(X_train_norm, y_train)

                        # Predict and score
                        y_pred = model.predict(X_val_norm)
                        score = score_function(y_val, y_pred)
                        scores.append(score)

                    # Calculate mean score
                    mean_score = np.mean(scores)
                    param_key = (norm_name, n_neighbors, metric, weight)
                    results[param_key] = mean_score

    return results