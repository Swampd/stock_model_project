#linear_models.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score


def run_linear_models(X, y, n_splits=5):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
        # (Add other linear models here if you ever want to)
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for name, model in models.items():
        aucs, f1s, recalls, accuracies = [], [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_pred_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

            report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
            aucs.append(roc_auc_score(y_te, y_pred_prob))
            f1s.append(report["1"]["f1-score"])
            recalls.append(report["1"]["recall"])
            accuracies.append(report["accuracy"])

        # ADD YOUR RESULTS TO THE LIST HERE
        results.append({
            "Model": name,
            "ROC-AUC": round(sum(aucs) / len(aucs), 3),
            "F1 (Class 1)": round(sum(f1s) / len(f1s), 3),
            "Recall (Class 1)": round(sum(recalls) / len(recalls), 3),
            "Accuracy": round(sum(accuracies) / len(accuracies), 3),
        })

    return results
