from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def train_evaluate_rf_ts(X_train, y_train, X_test, y_test):
    """Train Random Forest baseline for time series (Financial Data) and evaluate"""
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return clf, metrics

def train_evaluate_xgb_ts(X_train, y_train, X_test, y_test):
    """Train XGBoost baseline for time series (Financial Data) and evaluate"""
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return clf, metrics
