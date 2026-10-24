"""Subject-independent training and evaluation (Section 4.8)."""

import gc

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 10
RECONSTRUCTION_WEIGHT = 0.392


def scores(y_true, y_pred):
    """Accuracy, precision, recall and F1 with cognitive load as the positive class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def folds(y, groups, n_splits=N_SPLITS):
    """Stratified group K-fold: no subject appears in both the training and the test part."""
    return StratifiedGroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)


def lr_schedule(lr, decay=0.97, min_lr=5e-5):
    """Exponential learning rate annealing with a lower bound."""
    return lambda epoch: max(lr * decay ** epoch, min_lr)


def train_capsnet(x_train, y_train, epochs=20, batch_size=32, lr=1e-4, verbose=0):
    """Train the Efficient-CapsNet with the reconstruction regularizer and return the evaluation model."""
    import tensorflow as tf

    from .model import build_model, margin_loss
    from .model.capsnet import CAPSNET_NAME

    n_classes = 2
    y_onehot = tf.keras.utils.to_categorical(y_train, n_classes)
    train_model, eval_model = build_model(x_train.shape[1:], n_classes)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[margin_loss, "mse"],
        loss_weights=[1.0, RECONSTRUCTION_WEIGHT],
        metrics={CAPSNET_NAME: [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]},
    )
    train_model.fit(
        [x_train, y_onehot], [y_onehot, x_train],
        batch_size=batch_size, epochs=epochs, verbose=verbose,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule(lr))],
    )
    return eval_model


def predict_capsnet(model, x, batch_size=256):
    """Predicted class of each sample (the longest class capsule)."""
    class_scores, _ = model.predict(x, batch_size=batch_size, verbose=0)
    return np.argmax(class_scores, axis=1)


def cross_validate_capsnet(x, y, groups, n_splits=N_SPLITS, **train_kwargs):
    """Per-fold test scores of the capsule network."""
    import tensorflow as tf

    rows = []
    for i, (train_idx, test_idx) in enumerate(folds(y, groups, n_splits)):
        model = train_capsnet(x[train_idx], y[train_idx], **train_kwargs)
        rows.append({"fold": i, **scores(y[test_idx], predict_capsnet(model, x[test_idx]))})
        print(f"fold {i}: " + ", ".join(f"{k}={v:.4f}" for k, v in rows[-1].items() if k != "fold"))
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return pd.DataFrame(rows)


def summarize(df):
    """Mean and standard deviation of fold scores."""
    return df.drop(columns=[c for c in ("fold", "model") if c in df]).agg(["mean", "std"]).T
