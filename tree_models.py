# tree_models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesClassifier

def run_tree_models(X, y, n_splits=5):
    # Calculate class ratio for XGBoost (majority/minority)
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            scale_pos_weight=scale_pos_weight
        ),
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),  # <--- COMMA!
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    }

    # ...rest of your function as before...


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

        results.append({
            "Model": name,
            "ROC-AUC": round(sum(aucs) / len(aucs), 3),
            "F1 (Class 1)": round(sum(f1s) / len(f1s), 3),
            "Recall (Class 1)": round(sum(recalls) / len(recalls), 3),
            "Accuracy": round(sum(accuracies) / len(accuracies), 3),
        })

    return results
