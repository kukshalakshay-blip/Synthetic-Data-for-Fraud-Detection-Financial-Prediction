from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb

def train_evaluate_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest baseline and evaluate"""
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='macro'),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return clf, metrics

def train_evaluate_xgb(X_train, y_train, X_test, y_test):
    """Train XGBoost baseline and evaluate"""
    clf = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='macro'),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return clf, metrics
