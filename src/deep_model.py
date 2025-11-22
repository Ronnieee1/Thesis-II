import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import psutil
import time


def build_cnn_bilstm(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


class CPUPerEpochCallback(callbacks.Callback):
    """Keras callback to record CPU utilization and validation metrics at the end of each epoch.

    It samples CPU a few times at epoch end and computes validation metrics using provided validation
    data (X_val, y_val). X_val should be in the same shape expected by the Keras model (e.g. 3D).
    """
    def __init__(self, X_val=None, y_val=None, sample_interval=0.05, samples=3):
        super().__init__()
        self.sample_interval = sample_interval
        self.samples = samples
        self.cpu_per_epoch = []
        self.epoch_metrics = []
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # Take a few short CPU samples and average them to reduce noise
        vals = []
        for _ in range(self.samples):
            vals.append(psutil.cpu_percent(interval=self.sample_interval))
        avg_cpu = sum(vals) / len(vals)
        self.cpu_per_epoch.append(avg_cpu)

        # Compute validation metrics if validation data was provided
        if self.X_val is not None and self.y_val is not None:
            try:
                # Predict probabilities
                y_prob = self.model.predict(self.X_val).flatten()
            except Exception:
                # Fallback: try predicting on single-dim input
                y_prob = self.model.predict(self.X_val)
                import numpy as _np
                y_prob = _np.array(y_prob).flatten()

            # Convert to numpy arrays
            import numpy as _np
            y_true = _np.array(self.y_val).flatten()
            y_scores = _np.array(y_prob).flatten()

            # Compute classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            # For threshold-based metrics use 0.5
            y_pred = (y_scores >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y_true, y_scores))
            except Exception:
                auc = float('nan')

            metrics = {
                'epoch': int(epoch) + 1,
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'roc_auc': auc,
                'cpu_pct': float(avg_cpu)
            }
            self.epoch_metrics.append(metrics)


def train_model(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, patience=10):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    cpu_cb = CPUPerEpochCallback(X_val=X_val, y_val=y_val, sample_interval=0.02, samples=5)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, cpu_cb],
        verbose=1
    )
    return history, cpu_cb.cpu_per_epoch, cpu_cb.epoch_metrics
