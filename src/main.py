# Main orchestration script for adaptive hybrid phishing URL classification
import os
# Reduce TensorFlow oneDNN noise and lower TF log level (set before TF is imported)
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from data_loader import load_datasets
from feature_engineering import extract_lexical_features, clean_and_scale, apply_smote
from classical_model import chi_square_feature_selection, train_random_forest, evaluate_model
# Import deep model lazily later to ensure env vars are set before TF loads
from adaptive_fusion import adaptive_control, final_prediction
from evaluate import cross_validate_model, statistical_tests
from visualize import plot_learning_curves, plot_roc_curve, table_cpu_per_sample, plot_cpu_per_epoch, table_cpu_per_epoch

# Delay importing deep_model until after CSV path resolution (to avoid TF loading before env vars)
deep_model = None

# Step 1: Load dataset (single CSV) and select features
csv_path = r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\src/Dataset/src/Dataset/src/Dataset/Test1__lexical_standardized.csv"
import pandas as pd
import glob

# If the configured path doesn't exist, try to find a suitable CSV in the src/Dataset folder
if not os.path.exists(csv_path):
	dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset')
	print(f"Configured CSV path not found: {csv_path}\nSearching for candidates in {dataset_dir}...")
	candidates = glob.glob(os.path.join(dataset_dir, '*lexical*.csv')) + glob.glob(os.path.join(dataset_dir, '*.csv'))
	if candidates:
		csv_path = candidates[0]
		print(f"Using detected CSV: {csv_path}")
	else:
		raise FileNotFoundError(f"CSV not found at {csv_path} and no CSV candidates in {dataset_dir}. Please check the path.")

# read CSV
df = pd.read_csv(csv_path, low_memory=False)

# Expected lexical columns (from your list)
expected_cols = [
	"id","url","url_length","path_length","query_length","num_dots","num_underscores",
	"num_percent","num_ampersands","subdomain_count","has_https","http_in_hostname",
	"has_ip","num_special","contains_suspicious","domain_entropy","digit_count",
	"hyphen_count","has_at_symbol","double_slash_in_path","has_dash_prefix_suffix",
	"has_multi_subdomains","has_external_favicon","scheme","subdomain","domain","suffix",
	"netloc","path","query"
]

# Warn if some expected cols are missing
missing = [c for c in expected_cols if c not in df.columns]
if missing:
	print(f"Warning: missing expected columns in CSV: {missing}")

# Text fields (exclude from numeric features and from label auto-detection)
text_fields = {"id", "url", "scheme", "subdomain", "domain", "suffix", "netloc", "path", "query"}

# Detect label column (common names, env override), case-insensitive search, or auto-detect a binary column
import os
label_env = os.getenv('LABEL_COLUMN')
if label_env:
	if label_env in df.columns:
		label_col = label_env
	else:
		raise ValueError(f"Environment LABEL_COLUMN={label_env} but no such column in CSV.")
else:
	label_candidates = ["label", "CLASS_LABEL", "class", "is_phishing", "phishing", "target"]
	label_col = next((c for c in label_candidates if c in df.columns), None)
	if label_col is None:
		# try case-insensitive match for keywords
		lowered = {c.lower(): c for c in df.columns}
		for key in ['label', 'class', 'target', 'is_phish', 'phish']:
			if key in lowered:
				label_col = lowered[key]
				break

	if label_col is None:
		# heuristic: check for columns containing both 'class' and 'label' (e.g. 'Class_Label')
		for c in df.columns:
			low = c.lower()
			if 'class' in low and 'label' in low:
				label_col = c
				print(f"Detected label column by heuristic: {label_col}")
				break
		# attempt to auto-detect binary columns (excluding known feature/text columns)
		candidates = []
		for c in df.columns:
			if c in expected_cols or c in text_fields:
				continue
			vals = df[c].dropna().unique()
			if len(vals) == 0:
				continue
			# numeric 0/1 or boolean True/False
			try:
				vset = set(int(x) for x in vals)
				if vset.issubset({0, 1}):
					candidates.append(c)
					continue
			except Exception:
				pass
			# boolean strings or two distinct string values
			if len(vals) == 2:
				candidates.append(c)

		if len(candidates) == 1:
			label_col = candidates[0]
			print(f"Auto-detected label column: {label_col}")
		elif len(candidates) > 1:
			print("Multiple possible label columns detected:", candidates)
			raise ValueError("Multiple candidate label columns detected. Set the environment variable LABEL_COLUMN to the correct column name.")
		else:
			# No label found — try merging an external labels CSV if provided via LABELS_CSV
			labels_path = os.getenv('LABELS_CSV')
			if labels_path:
				print(f"LABELS_CSV provided: {labels_path} — attempting to merge labels into dataset")
				try:
					labdf = pd.read_csv(labels_path, low_memory=False)
					# prefer join on 'id' then 'url'
					if 'id' in df.columns and 'id' in labdf.columns:
						df = df.merge(labdf, on='id', how='left', suffixes=('','_labels'))
						print("Merged labels on 'id'")
					elif 'url' in df.columns and 'url' in labdf.columns:
						df = df.merge(labdf, on='url', how='left', suffixes=('','_labels'))
						print("Merged labels on 'url'")
					else:
						print("Labels file found but no 'id' or 'url' column to join on; merge skipped.")
					# look again for common label column names after merge
					for cand in ['label','CLASS_LABEL','class','is_phishing','phishing','target']:
						if cand in df.columns:
							label_col = cand
							break
					# handle possible suffixed label column like 'label_labels'
					if label_col is None:
						for c in df.columns:
							if c.lower().startswith(('label','class','target','is_phish','phish')) and c.endswith('_labels'):
								label_col = c
								break
				except Exception as e:
					print("Failed to read/merge LABELS_CSV:", e)

			# If still no label column, prompt the user interactively for next steps
			if label_col is None:
				print("\nNo label column found in the dataset.")
				# First, try to ask the user in the console
				try:
					choice = input("Enter label column name, path to labels CSV to merge, or 'unsupervised' to run IsolationForest (leave blank to abort): ").strip()
				except Exception:
					choice = None
				if choice:
					# if user provided a path to a file
					if os.path.exists(choice):
						try:
							labdf = pd.read_csv(choice, low_memory=False)
							if 'id' in df.columns and 'id' in labdf.columns:
								df = df.merge(labdf, on='id', how='left', suffixes=('','_labels'))
								print("Merged labels on 'id'")
							elif 'url' in df.columns and 'url' in labdf.columns:
								df = df.merge(labdf, on='url', how='left', suffixes=('','_labels'))
								print("Merged labels on 'url'")
							else:
								print("Labels file found but no 'id' or 'url' to join on; merge skipped.")
							# re-check for label column
							for cand in ['label','CLASS_LABEL','class','is_phishing','phishing','target']:
								if cand in df.columns:
									label_col = cand
									break
							# also check for suffixed labels
							if label_col is None:
								for c in df.columns:
									if c.lower().startswith(('label','class','target','is_phish','phish')) and c.endswith('_labels'):
										label_col = c
										break
						except Exception as e:
							print('Failed to read/merge provided labels file:', e)
					else:
						# treat input as column name
						if choice in df.columns:
							label_col = choice
							print(f"Using provided label column: {label_col}")
						elif choice.lower() == 'unsupervised':
							print('Proceeding with unsupervised IsolationForest (no true labels will be available).')
							# set a flag for unsupervised; downstream code may handle
							label_col = None
							unlabeled_mode = True
						else:
							print('Provided label column not found in dataset.')
				# If still no label_col after prompt, show message and exit
				if label_col is None:
					print('\nNo label column provided. Aborting. You can set env var LABELS_CSV or LABEL_COLUMN next time.')
					import sys
					sys.exit(1)


# Map label column values to 0/1 if necessary
# Ensure we don't rely on an 'unlabeled' flag anymore


# Map label column values to 0/1 if necessary
def map_labels_to_binary(series):
	s = series.dropna()
	# if numeric 0/1 already
	if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
		uniq = pd.unique(s)
		if set(uniq).issubset({0, 1}):
			return series.astype(int)
	# lowercase string mapping
	str_vals = [str(x).lower() for x in pd.unique(s)]
	positives = {'phishing', 'phish', 'malicious', '1', 'true', 'yes', 'y'}
	negatives = {'legitimate', 'legit', 'benign', '0', 'false', 'no', 'n'}
	# if one of the two values matches known positives/negatives
	if len(str_vals) == 2:
		v0, v1 = str_vals
		if v0 in positives and v1 in negatives:
			return series.apply(lambda x: 1 if str(x).lower() == v0 else 0).astype(int)
		if v1 in positives and v0 in negatives:
			return series.apply(lambda x: 1 if str(x).lower() == v1 else 0).astype(int)
		# default mapping: first unique -> 0, second -> 1
		unique_orig = list(pd.unique(s))
		mapping = {str(unique_orig[0]): 0, str(unique_orig[1]): 1}
		return series.apply(lambda x: mapping.get(str(x), 0)).astype(int)
	# fallback: try to coerce to numeric 0/1
	try:
		coerced = pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
		if set(pd.unique(coerced)).issubset({0, 1}):
			return coerced
	except Exception:
		pass
	# if all else fails, raise
	raise ValueError(f"Unable to map label column '{label_col}' to binary 0/1 values. Sample values: {pd.unique(s)[:10]}")

# Map and validate the label column (must exist at this point)
df[label_col] = map_labels_to_binary(df[label_col])

# Preserve id column (if present) for per-sample reporting
id_series = df['id'] if 'id' in df.columns else df.index

# Exclude long text fields from numeric features
text_fields = {"id", "url", "scheme", "subdomain", "domain", "suffix", "netloc", "path", "query"}
feature_cols = [c for c in expected_cols if c in df.columns and c not in text_fields]
if not feature_cols:
	# fallback: take all numeric columns except the label
	feature_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]

print(f"Using feature columns: {feature_cols}")

# Build X and y for supervised training
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df[label_col].astype(int)

# Split into train/validation sets (keep ids)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
	X, y, id_series, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Classical model (α-path)

# Classical model (α-path)
X_selected, selected_idx = chi_square_feature_selection(X_train, y_train)
rf_model = train_random_forest(X_selected, y_train)
import numpy as _np

# Validate label distribution in training/validation
train_classes, train_counts = _np.unique(y_train, return_counts=True)
val_classes, val_counts = _np.unique(y_val, return_counts=True)
print(f"Training label distribution: {dict(zip(train_classes, train_counts))}")
print(f"Validation label distribution: {dict(zip(val_classes, val_counts))}")
if len(train_classes) < 2:
	print("Warning: only one class present in training labels. The classifier will be trained for a single class.")
if len(val_classes) < 2:
	print("Warning: only one class present in validation labels. Metrics like AUC may be undefined.")

rf_metrics = evaluate_model(rf_model, X_val.iloc[:, selected_idx], y_val)

# Robustly extract probability for the positive class (1).
# sklearn's predict_proba returns shape (n_samples, n_classes). If only one class
# was present during training, it may return a single column — handle that.
try:
	proba = rf_model.predict_proba(X_val.iloc[:, selected_idx].values)
except Exception:
	proba = rf_model.predict_proba(X_val.iloc[:, selected_idx])

if proba.ndim == 1:
	# unexpected 1-D output; coerce
	proba = proba.reshape(-1, 1)

if proba.shape[1] == 1:
	trained_class = getattr(rf_model, 'classes_', [None])[0]
	if trained_class == 1:
		p_a = _np.ones(proba.shape[0])
	else:
		p_a = _np.zeros(proba.shape[0])
else:
	classes_list = list(getattr(rf_model, 'classes_', []))
	if 1 in classes_list:
		idx1 = classes_list.index(1)
		p_a = proba[:, idx1]
	else:
		# No class '1' in trained classes — fallback to last column as 'positive'
		p_a = proba[:, -1]

# Step 4: Deep model (β-path)

# Import deep_model now that env vars are set
from deep_model import build_cnn_bilstm, train_model

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

# Compute and print fused model metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

final_probs = np.array(p_final)
final_preds = (final_probs >= 0.5).astype(int)
try:
	auc = roc_auc_score(y_val, final_probs)
except Exception:
	auc = float('nan')

acc = accuracy_score(y_val, final_preds)
prec = precision_score(y_val, final_preds, zero_division=0)
rec = recall_score(y_val, final_preds, zero_division=0)
f1 = f1_score(y_val, final_preds, zero_division=0)

print('\nFused model evaluation (final prediction):')
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'AUC-ROC: {auc:.4f}\n')

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
# Ensure per-model tables are sorted by `id` (numeric when possible) for predictable ordering
try:
	df_rf['id_sort'] = pd.to_numeric(df_rf['id'], errors='coerce')
	df_rf = df_rf.sort_values(by='id_sort').drop(columns=['id_sort']).reset_index(drop=True)
except Exception:
	df_rf = df_rf.sort_values(by='id').reset_index(drop=True)

try:
	df_cnn['id_sort'] = pd.to_numeric(df_cnn['id'], errors='coerce')
	df_cnn = df_cnn.sort_values(by='id_sort').drop(columns=['id_sort']).reset_index(drop=True)
except Exception:
	df_cnn = df_cnn.sort_values(by='id').reset_index(drop=True)

# re-save sorted per-model CSVs
df_rf.to_csv(r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_table_rf.csv", index=False)
df_cnn.to_csv(r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_table_cnn.csv", index=False)

# Create a final per-sample table combining RF, CNN and fused predictions
import numpy as _np

# Prefer IDs returned by the per-sample tables (they use original dataset ids when provided)
if 'id' in df_rf.columns:
	ids_final = df_rf['id'].tolist()
elif len(ids_rf) == len(df_rf):
	ids_final = ids_rf
else:
	ids_final = list(range(1, len(df_rf) + 1))

cpu_avg = (_np.array(df_rf['cpu_pct']) + _np.array(df_cnn['cpu_pct'])) / 2.0

# include rf/cnn predicted labels as well as probabilities
df_final = pd.DataFrame({
	'id': ids_final,
	'cpu_pct_avg': cpu_avg,
	'rf_prob': df_rf['pred_prob'].values,
	'rf_label': df_rf['pred_label'].values,
	'cnn_prob': df_cnn['pred_prob'].values,
	'cnn_label': df_cnn['pred_label'].values,
	'final_prob': final_probs,
	'final_label': final_preds,
	'true_label': df_rf['true_label'].values
})

df_final['final_label_str'] = df_final['final_label'].apply(lambda x: 'phishing' if int(x) == 1 else 'legitimate')
df_final['rf_label_str'] = df_final['rf_label'].apply(lambda x: 'phishing' if int(x) == 1 else 'legitimate')
df_final['cnn_label_str'] = df_final['cnn_label'].apply(lambda x: 'phishing' if int(x) == 1 else 'legitimate')

out_final = r"C:\Users\Ron G\Documents\Thesis II\src\Dataset\cpu_table_final.csv"
df_final.to_csv(out_final, index=False)
print(f'Wrote final per-sample CPU+predictions table to: {out_final}')
print('\nSample of final per-sample table:')
# Sort final table by id (numeric if possible) so rows are in id order
try:
	df_final['id_sort'] = pd.to_numeric(df_final['id'], errors='coerce')
	df_final = df_final.sort_values(by='id_sort').drop(columns=['id_sort']).reset_index(drop=True)
except Exception:
	df_final = df_final.sort_values(by='id').reset_index(drop=True)

# write sorted final table
df_final.to_csv(out_final, index=False)
print(df_final.head(40).to_string(index=False))

# NOTE: Fill in the data loading, splitting, and actual workflow as needed.
