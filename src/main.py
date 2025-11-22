# Main orchestration script for adaptive hybrid phishing URL classification
from data_loader import load_datasets
from feature_engineering import extract_lexical_features, clean_and_scale, apply_smote
from classical_model import chi_square_feature_selection, train_random_forest, evaluate_model
from deep_model import build_cnn_bilstm, train_model
from adaptive_fusion import adaptive_control, final_prediction
from evaluate import cross_validate_model, statistical_tests
from visualize import plot_learning_curves, plot_roc_curve, table_cpu_per_sample, plot_cpu_per_epoch, table_cpu_per_epoch

# Step 1: Load datasets
paths = [r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\Phishing_Legitimate_full.csv"]
datasets = load_datasets(paths)

# Step 2: Feature engineering

df = datasets[0]
X = df.drop('CLASS_LABEL', axis=1)
y = df['CLASS_LABEL']

# Split into train/validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Classical model (α-path)

# Classical model (α-path)
X_selected, selected_idx = chi_square_feature_selection(X_train, y_train)
rf_model = train_random_forest(X_selected, y_train)
rf_metrics = evaluate_model(rf_model, X_val.iloc[:, selected_idx], y_val)
p_a = rf_model.predict_proba(X_val.iloc[:, selected_idx])[:, 1]

# Step 4: Deep model (β-path)

# Deep model (β-path)
cnn_bilstm = build_cnn_bilstm(input_shape=(X_train.shape[1], 1))
X_train_dl = X_train.values.reshape(-1, X_train.shape[1], 1)
X_val_dl = X_val.values.reshape(-1, X_val.shape[1], 1)
history, cpu_per_epoch, epoch_metrics = train_model(cnn_bilstm, X_train_dl, y_train, X_val_dl, y_val)
plot_learning_curves(history.history)
p_b = cnn_bilstm.predict(X_val_dl).flatten()

# Show CPU utilization and validation metrics per epoch for the deep model
plot_cpu_per_epoch(epoch_metrics)
df_cpu_epoch = table_cpu_per_epoch(epoch_metrics)
df_cpu_epoch.to_csv(r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_per_epoch_cnn.csv", index=False)

# Step 5: Adaptive fusion

# Adaptive fusion
val_loss_a = 1 - rf_model.score(X_val.iloc[:, selected_idx], y_val)
val_loss_b = history.history['val_loss'][-1]
alpha, beta = adaptive_control(0.5, 0.5, val_loss_a, val_loss_b)
p_final = final_prediction(alpha, beta, p_a, p_b)

# Step 6: Evaluation and visualization

# Evaluation and visualization
plot_roc_curve(y_val, p_final)
scores = cross_validate_model(rf_model, X_train.iloc[:, selected_idx], y_train)
print('Random Forest CV scores:', scores)

# Show CPU/per-sample tables for each model (this will iterate over validation samples and measure CPU briefly per sample)
# Prepare IDs for per-sample tables (use original dataframe indices as sample IDs)
X_val_selected = X_val.iloc[:, selected_idx]
ids_rf = list(X_val_selected.index)
df_rf = table_cpu_per_sample(rf_model, X_val_selected.reset_index(drop=True), y_val.reset_index(drop=True), model_kind='sklearn', id_series=ids_rf)

# For CNN use the validation indices as IDs
ids_cnn = list(X_val.index)
df_cnn = table_cpu_per_sample(cnn_bilstm, X_val.reset_index(drop=True), y_val.reset_index(drop=True), model_kind='keras', id_series=ids_cnn)
# Optionally save
df_rf.to_csv(r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_table_rf.csv", index=False)
df_cnn.to_csv(r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_table_cnn.csv", index=False)

# NOTE: Fill in the data loading, splitting, and actual workflow as needed.
