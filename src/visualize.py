import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time
import numpy as np

# ---------------------------
# 1. Learning Curves
# ---------------------------
def plot_learning_curves(history):
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# ---------------------------
# 2. ROC Curve
# ---------------------------
def plot_roc_curve(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# ---------------------------
# 3. Class Distribution Plot
# ---------------------------
def plot_class_distribution(y):
    """
    y: labels (0 = legitimate, 1 = phishing)
    """

    phishing_count = (y == 1).sum()
    legitimate_count = (y == 0).sum()

    plt.figure()
    sns.barplot(
        x=['Legitimate', 'Phishing'],
        y=[legitimate_count, phishing_count]
    )
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.show()

    print("Legitimate:", legitimate_count)
    print("Phishing:", phishing_count)


# ---------------------------
# 4. CPU Utilization Plot
# ---------------------------
def monitor_cpu_usage(duration=10, interval=0.5):
    cpu_values = []
    timestamps = []

    start = time.time()

    print("Monitoring CPU usage...")

    while time.time() - start < duration:
        cpu = psutil.cpu_percent(interval=interval)
        cpu_values.append(cpu)
        timestamps.append(time.time() - start)

    # Plot CPU utilization
    plt.figure()
    plt.plot(timestamps, cpu_values)
    plt.title(f'CPU Utilization (Duration: {duration}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('CPU %')
    plt.show()

    print("Average CPU Utilization:", sum(cpu_values) / len(cpu_values))


# ---------------------------
# 5. CPU + Prediction Table
# ---------------------------
def table_cpu_and_predictions(duration=10, interval=0.5, y_pred=None):
    """
    Shows CPU utilization and phishing predictions in a table.
    y_pred should be a list/array of model predictions (0 or 1)
    """

    timestamps = []
    cpu_values = []

    start = time.time()
    print("Collecting CPU usage...")

    # Collect CPU usage
    while time.time() - start < duration:
        cpu = psutil.cpu_percent(interval=interval)
        cpu_values.append(cpu)
        timestamps.append(round(time.time() - start, 2))

    # If predictions are not provided, generate dummy ones
    if y_pred is None:
        y_pred = np.random.randint(0, 2, len(cpu_values))

    # Convert prediction to readable text
    pred_labels = ["Phishing (1)" if p == 1 else "Legitimate (0)" for p in y_pred]

    # Prepare table data
    table_data = []
    for t, cpu, label in zip(timestamps, cpu_values, pred_labels):
        table_data.append([t, cpu, label])

    # Create the plot figure with table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("tight")
    ax.axis("off")

    table = plt.table(
        cellText=table_data,
        colLabels=["Time (s)", "CPU Usage (%)", "Prediction"],
        loc="center",
        cellLoc="center"
    )

    table.scale(1, 1.3)
    plt.title("CPU Usage and Phishing Predictions Table", fontsize=14)

    plt.show()

    print("Average CPU Utilization:", sum(cpu_values) / len(cpu_values))


def table_cpu_per_sample(model, X, y, model_kind='sklearn', threshold=0.5, interval=0.05, id_series=None):
    """
    Iterate through samples in X, record timestamp, CPU usage, prediction and true label.

    - model_kind: 'sklearn' for models with predict_proba, 'keras' for Keras models returning probabilities.
    - interval: psutil sampling interval in seconds used to get a short measurement per sample.

    Returns a pandas DataFrame with columns: ['time_s', 'cpu_pct', 'pred_prob', 'pred_label', 'true_label']
    """
    import pandas as pd

    records = []

    n = len(X)
    for i in range(n):
        # get a quick cpu reading (non-blocking short interval)
        cpu = psutil.cpu_percent(interval=interval)

        x_sample = X.iloc[i:i+1] if hasattr(X, 'iloc') else X[i:i+1]
        # use numpy arrays for sklearn/keras to avoid warnings about feature names
        x_sample_np = x_sample.values if hasattr(x_sample, 'values') else x_sample
        # assign IDs: prefer provided id_series (original dataset ids) if present
        if id_series is not None:
            try:
                # support pandas Series or list-like
                sample_id = id_series[i]
            except Exception:
                sample_id = i + 1
        else:
            # fallback sequential IDs (1..n)
            sample_id = i + 1

        if model_kind == 'sklearn':
            try:
                proba_arr = model.predict_proba(x_sample_np)
                # handle single-column proba outputs
                if proba_arr.ndim == 1:
                    proba_arr = proba_arr.reshape(-1, 1)
                if proba_arr.shape[1] == 1:
                    # if the model was trained on a single class, infer which class it is
                    classes = getattr(model, 'classes_', None)
                    if classes is not None and 1 in list(classes):
                        prob = float(proba_arr[:, 0][0])
                    else:
                        prob = float(0.0)
                else:
                    # try to select column for class '1'
                    classes = list(getattr(model, 'classes_', []))
                    if 1 in classes:
                        idx1 = classes.index(1)
                        prob = float(proba_arr[:, idx1][0])
                    else:
                        prob = float(proba_arr[:, -1][0])
            except Exception:
                prob = float(model.predict(x_sample_np)[0])
        else:
            # keras model expects 3D input for our CNN-BiLSTM
            x_in = x_sample_np.reshape(-1, x_sample.shape[1], 1)
            prob = float(model.predict(x_in).flatten()[0])

        pred_label = 1 if prob >= threshold else 0
        true_label = int(y.iloc[i]) if hasattr(y, 'iloc') else int(y[i])

        records.append([sample_id, cpu, prob, pred_label, true_label])

    df = pd.DataFrame(records, columns=['id', 'cpu_pct', 'pred_prob', 'pred_label', 'true_label'])

    # display as a table
    print('\nCPU usage and per-sample predictions:')
    print(df.head(200).to_string(index=False))

    # also show as a matplotlib table for convenience
    fig, ax = plt.subplots(figsize=(10, min(6, 0.25 * len(df) + 1)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title('CPU usage and per-sample predictions')
    plt.show()

    return df


def plot_cpu_per_epoch(cpu_list):
    """Plot CPU utilization per epoch as a line chart.

    Accepts either a list of cpu floats or a list of epoch-metric dicts (with key 'cpu_pct').
    """
    # If passed epoch_metrics (list of dicts), extract cpu values
    if len(cpu_list) > 0 and isinstance(cpu_list[0], dict):
        cpu_vals = [d.get('cpu_pct', None) for d in cpu_list]
    else:
        cpu_vals = cpu_list

    plt.figure()
    plt.plot(range(1, len(cpu_vals) + 1), cpu_vals, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Utilization per Epoch')
    plt.grid(True)
    plt.show()


def table_cpu_per_epoch(epoch_metrics):
    """Return and display a table (DataFrame) with epoch, accuracy, precision, recall, f1, roc_auc and cpu_pct.

    `epoch_metrics` should be a list of dicts produced by the training callback where each dict contains
    keys: 'epoch','accuracy','precision','recall','f1','roc_auc','cpu_pct'.
    """
    import pandas as pd

    if len(epoch_metrics) == 0:
        print('No epoch metrics available')
        return pd.DataFrame()

    # Normalize dicts into DataFrame
    df = pd.DataFrame(epoch_metrics)

    # Ensure column order
    cols = ['epoch', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cpu_pct']
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df[cols]

    print('\nCPU and validation metrics per epoch:')
    print(df.to_string(index=False))

    # show as matplotlib table
    fig, ax = plt.subplots(figsize=(10, min(2 + 0.4 * len(df), 10)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title('CPU and Validation Metrics per Epoch')
    plt.show()

    return df
