from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel, wilcoxon

def cross_validate_model(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    return scores

def statistical_tests(results_a, results_b):
    t_stat, t_p = ttest_rel(results_a, results_b)
    w_stat, w_p = wilcoxon(results_a, results_b)
    return {'t_test': (t_stat, t_p), 'wilcoxon': (w_stat, w_p)}
