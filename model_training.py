#model_training.py



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import start_date, end_date, test_holdout_ratio, drop_column_null_threshold, ALWAYS_KEEP, FEATURE_CORR_THRESHOLD, features_list
import csv

def drop_correlated_features(df, features, threshold=FEATURE_CORR_THRESHOLD):
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                to_drop.add(col)
    final_features = [f for f in features if f not in to_drop]
    print(f"\nDropped {len(to_drop)} highly correlated features (>{threshold}):\n{sorted(list(to_drop))}")
    print(f"Remaining features after deduplication: {len(final_features)}")
    return final_features



def run_model_training(data, features, label="future_up_3d", tied_thresh=0.50):
    print("\nNumber of rows BEFORE dropna:", len(data))
    print("Number of rows with NO NaNs in features+label:",
          data[features + [label]].dropna().shape[0])

    model_data = data[data[label].notnull()].copy()
    print(f"\nModeling on {len(model_data)} rows after dropna.")

    if len(model_data) == 0:
        print("\nERROR: No rows left after dropping NaNs! Check which features are causing the issue above.")
        print("\nNaN count per column:")
        print(data[features + [label]].isnull().sum())
        exit(1)

    model_data = model_data.sort_values("Date").reset_index(drop=True)

    holdout_frac = test_holdout_ratio  # change in config
    n_holdout = int(len(model_data) * holdout_frac)
    train_data = model_data.iloc[:-n_holdout]
    holdout_data = model_data.iloc[-n_holdout:]

    print(f"\nTraining on {len(train_data)} rows, holding out {len(holdout_data)} rows (from {holdout_data['Date'].min()} to {holdout_data['Date'].max()})")

    # --- FEATURE IMPORTANCE AND AUTO-DROP ---
    def cross_val_feature_importance(X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rf_importances = []
        xgb_importances = []
        lgbm_importances = []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_tr.fillna(X_tr.median()), y_tr)
            rf_importances.append(rf.feature_importances_)

            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                                scale_pos_weight=(y_tr.value_counts()[0] / y_tr.value_counts()[1]))
            xgb.fit(X_tr, y_tr)
            xgb_importances.append(xgb.feature_importances_)

            lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
            lgbm.fit(X_tr, y_tr)
            lgbm_importances.append(lgbm.feature_importances_)

        rf_mean = np.mean(rf_importances, axis=0)
        xgb_mean = np.mean(xgb_importances, axis=0)
        lgbm_mean = np.mean(lgbm_importances, axis=0)

        return (
            pd.Series(rf_mean, index=X.columns).sort_values(ascending=False),
            pd.Series(xgb_mean, index=X.columns).sort_values(ascending=False),
            pd.Series(lgbm_mean, index=X.columns).sort_values(ascending=False)
        )

    X_train_tmp = train_data[features].reset_index(drop=True)
    y_train_tmp = train_data[label].reset_index(drop=True)
    rf_imp, xgb_imp, lgbm_imp = cross_val_feature_importance(X_train_tmp, y_train_tmp, n_splits=5)

    print("\nCV-Averaged Random Forest Importances:")
    print(rf_imp)
    print("\nCV-Averaged XGBoost Importances:")
    print(xgb_imp)
    print("\nCV-Averaged LightGBM Importances:")
    print(lgbm_imp)

    lgbm_relative_thresh = tied_thresh * 100
    rf_thresh = tied_thresh
    xgb_thresh = tied_thresh
    lgbm_thresh = lgbm_relative_thresh

    weak_feats = set(rf_imp[rf_imp < rf_thresh].index) & \
                 set(xgb_imp[xgb_imp < xgb_thresh].index) & \
                 set(lgbm_imp[lgbm_imp < lgbm_thresh].index)

    print(f"\n=== Feature Drop Summary ===")
    print(f"Started with {len(features)} features.")
    print(f"Dropped by model importance: {len(weak_feats)} features:\n{sorted(list(weak_feats))}")

    reduced_features = [f for f in features if f not in weak_feats]
    print(f"Remaining features after model cull: {len(reduced_features)}")

    # --- Correlation culling here ---
    final_features = drop_correlated_features(model_data, reduced_features, threshold=FEATURE_CORR_THRESHOLD)

    # ==== Create training and holdout sets with pruned features ====
    X_train = train_data[final_features].reset_index(drop=True)
    y_train = train_data[label].reset_index(drop=True)
    X_holdout = holdout_data[final_features].reset_index(drop=True)
    y_holdout = holdout_data[label].reset_index(drop=True)

    # --- NaN Filling for SKLEARN models ---
    imputer = SimpleImputer(strategy="median")
    X_train_filled = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_holdout_filled = pd.DataFrame(imputer.transform(X_holdout), columns=X_holdout.columns, index=X_holdout.index)


    from sklearn.exceptions import NotFittedError

    # --- MODEL ZOO ---
    results = []

    def safe_roc_auc(y_true, proba):
        try:
            return roc_auc_score(y_true, proba)
        except Exception:
            return None

    # 1. RandomForestClassifier (sklearn, uses filled)
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train_filled, y_train)
    y_pred = rf.predict(X_holdout_filled)
    try:
        probs = rf.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "Random Forest",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[Random Forest]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 2. ExtraTreesClassifier (sklearn, uses filled)
    et = ExtraTreesClassifier(random_state=42, class_weight='balanced')
    et.fit(X_train_filled, y_train)
    y_pred = et.predict(X_holdout_filled)
    try:
        probs = et.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "Extra Trees",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[Extra Trees]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 3. GradientBoostingClassifier (sklearn, uses filled)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_filled, y_train)
    y_pred = gb.predict(X_holdout_filled)
    try:
        probs = gb.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "Gradient Boosting (sklearn)",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[Gradient Boosting (sklearn)]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 4. LogisticRegression (sklearn, uses filled)
    logr = LogisticRegression(max_iter=200, class_weight='balanced')
    logr.fit(X_train_filled, y_train)
    y_pred = logr.predict(X_holdout_filled)
    try:
        probs = logr.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "Logistic Regression",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[Logistic Regression]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 5. SVC (sklearn, uses filled)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_train_filled, y_train)
    y_pred = svc.predict(X_holdout_filled)
    try:
        probs = svc.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "SVM",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[SVM]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 6. KNeighborsClassifier (sklearn, uses filled)
    knn = KNeighborsClassifier()
    knn.fit(X_train_filled, y_train)
    y_pred = knn.predict(X_holdout_filled)
    try:
        probs = knn.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "KNN",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[KNN]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 7. XGBoost (NaN-tolerant)
    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_holdout)
    try:
        probs = xgb.predict_proba(X_holdout)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "XGBoost",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[XGBoost]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 8. LightGBM (NaN-tolerant)
    lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_holdout)
    try:
        probs = lgbm.predict_proba(X_holdout)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "LightGBM",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[LightGBM]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    # 9. MLPClassifier (sklearn neural net, uses filled)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train_filled, y_train)
    y_pred = mlp.predict(X_holdout_filled)
    try:
        probs = mlp.predict_proba(X_holdout_filled)[:, 1]
        roc_auc = safe_roc_auc(y_holdout, probs)
    except Exception:
        roc_auc = None
    results.append({
        "Model": "MLP Neural Net",
        "Report": classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"], output_dict=True),
        "ROC-AUC": roc_auc,
    })
    print("\n[MLP Neural Net]")
    print(classification_report(y_holdout, y_pred, target_names=["Not Up", "Up"]))

    MIN_COVERAGE = drop_column_null_threshold
    # ALWAYS_KEEP see config.py

    # Recalculate feature_coverage and to_drop for the audit log
    feature_coverage = 1 - data[features].isnull().mean()
    to_drop = [col for col, cov in feature_coverage.items() if (cov < MIN_COVERAGE and col not in ALWAYS_KEEP)]
    
    # --- Improved Feature Audit Log ---
    audit_rows = []
    all_features_set = set(features_list)
    final_features_set = set(final_features)
    dropped_by_cov = set([col for col in features_list if col not in features])
    dropped_by_model = set([col for col in features if col not in reduced_features])
    dropped_by_corr = set([col for col in reduced_features if col not in final_features])

    for col in sorted(all_features_set):
        # Start with all features you intended to use, sorted
        cov = feature_coverage.get(col, 0)
        status = ""
        reason = ""
        if col in final_features_set:
            status = "✅"
            reason = "IN MODEL"
        elif col in dropped_by_corr:
            status = "❌"
            reason = "Dropped: Correlated"
        elif col in dropped_by_model:
            status = "❌"
            reason = "Dropped: Model Cull"
        elif col in dropped_by_cov:
            status = "❌"
            reason = "Dropped: Low Coverage"
        elif col not in data.columns:
            status = "❌"
            reason = "Missing in Data"
        else:
            status = "❌"
            reason = "Other"
        audit_rows.append({
            "Feature": col,
            "Coverage %": f"{cov:.1%}",
            "Status": status,
            "Reason": reason
        })

    print("\n=== Feature Audit Log ===")
    print("{:<28} {:<10} {:<3} {:<20}".format("Feature", "Coverage", "", "Reason"))
    for row in audit_rows:
        print("{:<28} {:<10} {:<3} {:<20}".format(
            row["Feature"], row["Coverage %"], row["Status"], row["Reason"]
        ))

    # Optionally write to CSV for Excel
    import csv
    with open("feature_audit_log.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Feature", "Coverage %", "Status", "Reason"])
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)
    print("\nFeature audit log saved to feature_audit_log.csv")


    # === Pretty Summary Block ===
    print("\n=== Model Cross-Validation Summary ===")
    print(f"Cull Value: {tied_thresh}, Dates: {start_date} - {end_date}")
    print("{:<28} {:<10} {:<10} {:<10} {:<10}".format("Model", "ROC-AUC", "F1 (Up)", "Accuracy", "Recall (Up)"))

    for res in results:
        roc_auc = res.get("ROC-AUC")
        if roc_auc is None:
            roc_auc = "-"
        else:
            roc_auc = f"{roc_auc:.3f}"
        f1_up = res["Report"]["Up"]["f1-score"]
        accuracy = res["Report"]["accuracy"]
        recall_up = res["Report"]["Up"]["recall"]
        print("{:<28} {:<10} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            res['Model'], roc_auc, f1_up, accuracy, recall_up
        ))

    print("\nAll model runs complete.\n")

    return

