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
    y_pred = clf.predict(X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred)
    }
