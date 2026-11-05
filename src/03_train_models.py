"""
STEP 3 - Machine learning: forecast the anchorage queue H hours ahead.

Validation = rolling origin (expanding window, one day per fold): train on everything up to day D,
predict day D+1, repeat. This is a strictly time-based split - the model never sees the future.

Models compared (all predict the CHANGE from the current queue, then add it back):
  baselines  : "same as now" (persistence), "average of last 24 h"
  Linear     : Ridge-regularised linear regression
  RandomForest, LightGBM
  Final      : equal-weight average of Ridge + Random Forest + LightGBM (an ensemble is more stable on 2 weeks of data)
Congestion class (Low / Medium / High): terciles of the training queue, re-computed per fold.

Output: outputs/models/*.joblib, outputs/tables/model_results.csv, classification_results.csv,
        predictions_oof.csv, shap_importance_h24.csv, outputs/charts/shap_summary.png
"""
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config as C
from common import congestion_class, congestion_thresholds
from features import ALL_FEATURES, CALENDAR, CORE, FEATURES, FEATURE_LABELS, PORT_STATE, WEATHER, build, load_hourly

SEED = C.RANDOM_STATE


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=3.0))


def make_rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)


def make_lgb():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=5, min_child_samples=8,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, verbose=-1)


def make_lgb_clf():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=5, min_child_samples=8,
                              class_weight="balanced", random_state=SEED, verbose=-1)


def make_rf_clf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def make_lr_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000))


CLF_FACTORY = {"LightGBM classifier": make_lgb_clf, "Random Forest classifier": make_rf_clf,
               "Logistic Regression classifier": make_lr_clf}


def rolling_origin(f, horizon):
    """Yield (train, test) with an expanding training window and one-day test folds."""
    n = len(f)
    for start in range(C.MIN_TRAIN_ROWS, n - 6, C.FOLD_HOURS):
        # leave a gap of `horizon` hours so no training target overlaps the test period
        train = f.iloc[:max(start - horizon, 1)]
        test = f.iloc[start:start + C.FOLD_HOURS]
        if len(train) >= 48 and len(test):
            yield train, test


def evaluate_horizon(f, horizon, cols=None):
    cols = cols or FEATURES
    recs = []
    for k, (tr, te) in enumerate(rolling_origin(f, horizon)):
        d = pd.DataFrame(index=te.index)
        d["fold"] = k
        d["actual"] = te.y
        d["persistence"] = te.queue_now
        d["mean24"] = te.queue_mean_24h
        delta = tr.y - tr.queue_now
        d["ridge"] = te.queue_now + make_ridge().fit(tr[cols], delta).predict(te[cols])
        d["rf"] = te.queue_now + make_rf().fit(tr[cols], delta).predict(te[cols])
        d["lgbm"] = te.queue_now + make_lgb().fit(tr[cols], delta).predict(te[cols])
        d["blend"] = (d.ridge + d.rf + d.lgbm) / 3
        q1, q2 = congestion_thresholds(tr.y)
        d["q1"], d["q2"] = q1, q2
        d["actual_class"] = congestion_class(te.y, q1, q2)
        d["persistence_class"] = congestion_class(te.queue_now, q1, q2)
        d["blend_class"] = congestion_class(d.blend, q1, q2)
        lab_tr = congestion_class(tr.y, q1, q2)
        for name, mk in (("lgbm_clf", make_lgb_clf), ("rf_clf", make_rf_clf), ("logit_clf", make_lr_clf)):
            if len(set(lab_tr)) == 3:
                d[name] = mk().fit(tr[cols], lab_tr).predict(te[cols])
            else:
                d[name] = d.persistence_class
        recs.append(d)
    return pd.concat(recs)


def main():
    h = load_hourly()
    reg_rows, cls_rows, oof_all = [], [], []
    for H in C.HORIZONS:
        f = build(h, H)
        oof = evaluate_horizon(f, H)
        oof["horizon_h"] = H
        oof_all.append(oof)
        y = oof.actual.values
        base_mae = mean_absolute_error(y, oof.persistence)
        names = {"persistence": "Baseline: same as now", "mean24": "Baseline: average of last 24 h",
                 "ridge": "Linear Regression (Ridge)", "rf": "Random Forest", "lgbm": "LightGBM",
                 "blend": "Final: Ridge + RF + LightGBM blend"}
        for k, label in names.items():
            p = oof[k].values
            reg_rows.append(dict(horizon_h=H, model=label, n_test_hours=len(y),
                                 MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5,
                                 R2=r2_score(y, p), skill_vs_persistence_pct=(1 - mean_absolute_error(y, p) / base_mae) * 100))
        order = ["Low", "Medium", "High"]
        ya = oof.actual_class.values
        for k, label in {"persistence_class": "Baseline: class of queue now", "blend_class": "Class from blend forecast",
                         "lgbm_clf": "LightGBM classifier", "rf_clf": "Random Forest classifier",
                         "logit_clf": "Logistic Regression classifier"}.items():
            p = oof[k].values
            cls_rows.append(dict(horizon_h=H, model=label, n_test_hours=len(ya), n_high_hours=int((ya == "High").sum()),
                                 macro_F1=f1_score(ya, p, average="macro", labels=order, zero_division=0),
                                 recall_High=recall_score(ya == "High", p == "High", zero_division=0),
                                 precision_High=precision_score(ya == "High", p == "High", zero_division=0),
                                 high_caught=int(((p == "High") & (ya == "High")).sum()),
                                 false_alarms=int(((p == "High") & (ya != "High")).sum())))
        print(f"H={H:>2}h  persistence MAE {base_mae:.2f} | blend MAE "
              f"{mean_absolute_error(y, oof.blend):.2f} | test hours {len(y)}")

    res = pd.DataFrame(reg_rows).round(3)
    cls = pd.DataFrame(cls_rows).round(3)
    res.to_csv(C.TABLES / "model_results.csv", index=False)
    cls.to_csv(C.TABLES / "classification_results.csv", index=False)
    pd.concat(oof_all).reset_index().rename(columns={"hour_utc": "origin_utc"}).to_csv(
        C.TABLES / "predictions_oof.csv", index=False)
    print("\n", res[res.horizon_h == C.FINAL_HORIZON].to_string(index=False))
    print("\n", cls[cls.horizon_h == C.FINAL_HORIZON].to_string(index=False))

    # ---------------- feature ablation: do port state / calendar / weather help? ----------------
    sets = {"Queue history + time of day (final)": CORE,
            "+ berth occupancy & arrivals": CORE + PORT_STATE,
            "+ weekday": CORE + CALENDAR,
            "+ weather (wave, wind)": CORE + WEATHER,
            "Everything": ALL_FEATURES}
    abl = []
    for H in C.HORIZONS:
        f = build(h, H)
        for name, cols in sets.items():
            o = evaluate_horizon(f, H, cols)
            abl.append(dict(horizon_h=H, feature_set=name, n_features=len(cols),
                            MAE_blend=mean_absolute_error(o.actual, o.blend),
                            MAE_persistence=mean_absolute_error(o.actual, o.persistence)))
    abl = pd.DataFrame(abl).round(3)
    abl.to_csv(C.TABLES / "feature_ablation.csv", index=False)
    print("\nFEATURE ABLATION (MAE of the blend, lower is better)")
    print(abl.pivot(index="feature_set", columns="horizon_h", values="MAE_blend").to_string())

    # ---------------- final models: train on ALL labelled hours, one set per horizon ----------------
    for H in C.HORIZONS:
        f = build(h, H)
        q1, q2 = congestion_thresholds(f.y)
        delta = f.y - f.queue_now
        ridge = make_ridge().fit(f[FEATURES], delta)
        rf = make_rf().fit(f[FEATURES], delta)
        lgbm = make_lgb().fit(f[FEATURES], delta)
        # final classifier = the one with the best macro-F1 in rolling-origin validation at this horizon
        cand = cls[(cls.horizon_h == H) & cls.model.isin(CLF_FACTORY)].sort_values("macro_F1", ascending=False)
        best_clf = cand.iloc[0].model
        clf = CLF_FACTORY[best_clf]().fit(f[FEATURES], congestion_class(f.y, q1, q2))
        r = res[(res.horizon_h == H)].set_index("model")
        joblib.dump(dict(horizon_h=H, features=FEATURES, ridge=ridge, rf=rf, lgbm=lgbm, classifier=clf, classifier_name=best_clf,
                         q1=q1, q2=q2, trained_on_hours=len(f), last_origin=str(f.index.max()),
                         cv_mae_blend=float(r.loc["Final: Ridge + RF + LightGBM blend", "MAE"]),
                         cv_mae_persistence=float(r.loc["Baseline: same as now", "MAE"])),
                    C.MODELS / f"queue_forecast_{H}h.joblib")
    print("\nSaved final models for horizons", C.HORIZONS)

    # ---------------- SHAP for the headline horizon (LightGBM part) ----------------
    H = C.FINAL_HORIZON
    f = build(h, H)
    m = make_lgb().fit(f[FEATURES], f.y - f.queue_now)
    X = f[FEATURES].rename(columns=FEATURE_LABELS)
    sv = shap.TreeExplainer(m).shap_values(f[FEATURES])
    imp = pd.DataFrame({"feature": [FEATURE_LABELS[c] for c in FEATURES],
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TABLES / f"shap_importance_h{H}.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=10)
    plt.title(f"What drives the {H}-hour-ahead queue forecast (SHAP)")
    plt.tight_layout()
    plt.savefig(C.CHARTS / "shap_summary.png", dpi=150)
    plt.close()
    print(imp.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
