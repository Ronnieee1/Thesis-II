from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def chi_square_feature_selection(X, y, k=20):
    # FIX: chi2 requires all features to be non-negative.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X_scaled, y)
    return X_new, selector.get_support(indices=True)

def train_random_forest(X, y):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

def evaluate_model(clf, X, y):
    # Ensure we pass numpy arrays to sklearn to avoid feature-name warnings
    try:
        X_in = X.values if hasattr(X, 'values') else X
    except Exception:
        X_in = X

    y_pred = clf.predict(X_in)
    # For roc_auc, prefer probability estimates when available
    try:
        y_score = clf.predict_proba(X_in)[:, 1]
    except Exception:
        # fallback to label predictions
        y_score = y_pred

    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_score) if len(set(y)) > 1 else float('nan')
    }
