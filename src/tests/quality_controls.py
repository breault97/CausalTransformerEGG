import numpy as np


def test_permutation(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Permute labels on train set and ensure performance drops to chance.
    Returns dict with balanced accuracy and macro-F1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_perm = np.random.permutation(y_train)
    clf = LogisticRegression(max_iter=200, n_jobs=1, multi_class="auto")
    clf.fit(X_train, y_perm)
    y_pred = clf.predict(X_test)
    return {
        "perm_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "perm_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def test_subject_prediction(X_train: np.ndarray, subj_train: np.ndarray,
                            X_test: np.ndarray, subj_test: np.ndarray):
    """
    Predict subject IDs from raw features.
    High accuracy indicates subject-specific leakage.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, n_jobs=1, multi_class="auto")
    clf.fit(X_train, subj_train)
    y_pred = clf.predict(X_test)
    return {"subject_id_acc": float(np.mean(y_pred == subj_test))}


def test_channel_ablation(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          input_channels: int, input_samples: int,
                          ablate_channels):
    """
    Zero selected channels and compare performance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    X_train_r = X_train.reshape(X_train.shape[0], input_channels, input_samples)
    X_test_r = X_test.reshape(X_test.shape[0], input_channels, input_samples)
    X_train_r[:, ablate_channels, :] = 0.0
    X_test_r[:, ablate_channels, :] = 0.0

    clf = LogisticRegression(max_iter=200, n_jobs=1, multi_class="auto")
    clf.fit(X_train_r.reshape(X_train.shape[0], -1), y_train)
    y_pred = clf.predict(X_test_r.reshape(X_test.shape[0], -1))
    return {
        "ablation_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "ablation_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
