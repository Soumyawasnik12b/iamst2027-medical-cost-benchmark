"""
IAMST 2027 benchmark: Gamma GLM vs tree ensembles for medical-charge prediction,
exact Shapley explanations, explanation stability and subgroup error audit.

Data: insurance.csv (Medical Cost Personal dataset, n = 1,338).
Requirements: numpy, pandas, scikit-learn, matplotlib.
Usage: python analysis.py insurance.csv            (scikit-learn gradient boosting; reproduces the paper)
       python analysis.py insurance.csv --xgboost  (optional XGBoost variant; results will differ)
Exact duplicate records are kept in the same fold in every cross-validation split.
Prespecified selection criterion: lowest mean repeated 5-fold CV RMSE on the 80% training portion
(RMSE targets the conditional mean, which is the estimand).
"""
import json, itertools, math, os, sys, warnings, hashlib
import numpy as np, pandas as pd, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, GammaRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
OUT = "out"; os.makedirs(OUT, exist_ok=True)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA = ARGS[0] if ARGS else "ds/insurance.csv"
USE_XGB = "--xgboost" in sys.argv
if USE_XGB:
    from xgboost import XGBRegressor
CONFIG_VERSION = "v2-grouped"

raw_bytes = open(DATA, "rb").read()
norm_bytes = raw_bytes.replace(b"\r\n", b"\n").rstrip(b"\n")   # line-ending-independent content hash
HASH_CONTENT = hashlib.sha256(norm_bytes).hexdigest()
HASH_RAW = hashlib.sha256(raw_bytes).hexdigest()
print("content SHA-256 (line-ending independent):", HASH_CONTENT)

df = pd.read_csv(DATA)
FEATS = ["age", "sex", "bmi", "children", "smoker", "region"]
CAT, NUM = ["sex", "smoker", "region"], ["age", "bmi", "children"]
X, y = df[FEATS], df["charges"].to_numpy()
# Exact duplicates share a group id so they are never split across training and validation folds.
GROUPS = df.groupby(FEATS + ["charges"], sort=False).ngroup().to_numpy()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df["smoker"])
assert not set(GROUPS[idx_tr]) & set(GROUPS[idx_te]), "duplicate records split across train/test"
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
gtr = GROUPS[idx_tr]

def group_splits(groups, n_splits=5, n_repeats=1, seed=SEED):
    """Repeated, shuffled K-fold that keeps each group (duplicate set) within one fold.
    Identical to scikit-learn's GroupKFold(shuffle=True, random_state=...) (>= 1.6),
    implemented here so the script gives the same folds on any scikit-learn version."""
    groups = np.asarray(groups); out = []
    for r in range(n_repeats):
        ug = np.random.RandomState(seed + r).permutation(np.unique(groups))
        for test_ids in np.array_split(ug, n_splits):
            test_mask = np.isin(groups, test_ids)
            out.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return out

def prep(scale=True):
    return ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT),
                              ("num", StandardScaler() if scale else "passthrough", NUM)])

def actuarial_terms(Z):
    Z = Z.copy(); s = (Z["smoker"] == "yes").astype(float)
    Z["age2"] = Z["age"] ** 2; Z["obese"] = (Z["bmi"] >= 30).astype(float)
    Z["smk_obese"] = s * Z["obese"]; Z["smk_bmi"] = s * Z["bmi"]
    return Z
EXTRA = ["age2", "obese", "smk_obese", "smk_bmi"]

class LogOLSSmearing(BaseEstimator, RegressorMixin):
    """OLS on log(charges), retransformed with Duan's smearing estimator."""
    def fit(self, X, y):
        self.p_ = Pipeline([("p", prep()), ("m", LinearRegression())]).fit(X, np.log(y))
        self.smear_ = np.mean(np.exp(np.log(y) - self.p_.predict(X))); return self
    def predict(self, X): return np.exp(self.p_.predict(X)) * self.smear_

fixed = {
    "OLS": Pipeline([("p", prep()), ("m", LinearRegression())]),
    "Gamma GLM": Pipeline([("p", prep()), ("m", GammaRegressor(alpha=0.0, max_iter=2000))]),
    "Gamma GLM + actuarial terms": Pipeline([("fe", FunctionTransformer(actuarial_terms)),
        ("p", ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT),
                                 ("num", StandardScaler(), NUM + EXTRA)])),
        ("m", GammaRegressor(alpha=0.0, max_iter=2000))]),
    "Log-OLS + smearing (sensitivity)": LogOLSSmearing(),
}
tuned = {
    "Decision tree": (Pipeline([("p", prep(False)), ("m", DecisionTreeRegressor(random_state=SEED))]),
        {"m__max_depth": [3, 4, 5, 6], "m__min_samples_leaf": [5, 10, 20]}),
    "Random forest": (Pipeline([("p", prep(False)), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
        {"m__max_depth": [4, 6, 8, None], "m__min_samples_leaf": [1, 5, 10], "m__max_features": [0.5, 1.0]}),
}
if USE_XGB:
    tuned["XGBoost"] = (Pipeline([("p", prep(False)), ("m", XGBRegressor(random_state=SEED, n_jobs=1, subsample=0.8))]),
        {"m__n_estimators": [100, 200, 400], "m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3, 4],
         "m__min_child_weight": [5, 10], "m__reg_lambda": [1, 10]})
    BOOST = "XGBoost"
else:
    tuned["Gradient boosting"] = (Pipeline([("p", prep(False)), ("m", GradientBoostingRegressor(random_state=SEED, subsample=0.8))]),
        {"m__n_estimators": [100, 200, 400], "m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3, 4],
         "m__min_samples_leaf": [5, 10]})
    BOOST = "Gradient boosting"

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
R = {"n": len(df), "ntr": len(Xtr), "nte": len(Xte), "boost_name": BOOST, "uses_xgboost": USE_XGB,
     "hash_content_sha256": HASH_CONTENT, "hash_raw_sha256": HASH_RAW, "config": CONFIG_VERSION, "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
     "duplicate_groups": int((pd.Series(GROUPS).value_counts() > 1).sum())}

# ---------- 1. Descriptives ----------
R["desc"] = dict(min=float(y.min()), median=float(np.median(y)), mean=float(y.mean()), max=float(y.max()),
                 sd=float(y.std(ddof=1)), skew=float(pd.Series(y).skew()), missing=int(df.isna().sum().sum()),
                 duplicates=int(df.duplicated().sum()), age=[int(df.age.min()), int(df.age.max())],
                 bmi=[float(df.bmi.min()), float(df.bmi.max()), float(df.bmi.mean())],
                 children=[int(df.children.min()), int(df.children.max())],
                 counts={c: df[c].value_counts().to_dict() for c in CAT},
                 mean_by={c: df.groupby(c).charges.mean().round(0).to_dict() for c in CAT},
                 test_smokers=int((Xte.smoker == "yes").sum()))

# ---------- 2. Tuning on training portion only ----------
models, R["best_params"], R["grid_size"] = dict(fixed), {}, {}
for name, (pipe, grid) in tuned.items():
    gs = GridSearchCV(pipe, grid, cv=group_splits(gtr),
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
    models[name] = gs.best_estimator_
    R["best_params"][name] = {k[3:]: v for k, v in gs.best_params_.items()}
    R["grid_size"][name] = len(gs.cv_results_["params"])
    print("tuned", name, R["best_params"][name], flush=True)

# ---------- 3. Repeated CV within training portion (selection) + single test evaluation ----------
rkf = group_splits(gtr, 5, 5, SEED)
perf = []
for name, m in models.items():
    cv = cross_validate(m, Xtr, ytr, cv=rkf, n_jobs=-1,
                        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    m.fit(Xtr, ytr); p = m.predict(Xte)
    top = yte >= np.quantile(yte, 0.9)
    perf.append(dict(model=name,
        cv_mae=-cv["test_neg_mean_absolute_error"].mean(), cv_mae_sd=cv["test_neg_mean_absolute_error"].std(),
        cv_rmse=-cv["test_neg_root_mean_squared_error"].mean(), cv_rmse_sd=cv["test_neg_root_mean_squared_error"].std(),
        cv_r2=cv["test_r2"].mean(), cv_r2_sd=cv["test_r2"].std(),
        te_mae=mean_absolute_error(yte, p), te_rmse=rmse(yte, p), te_r2=r2_score(yte, p),
        te_top10_mae=mean_absolute_error(yte[top], p[top]), te_balance=p.sum() / yte.sum()))
    print("perf", name, flush=True)
perf = pd.DataFrame(perf); print(perf.round(3).to_string())
R["perf"] = perf.to_dict("records")
cand = perf[~perf.model.str.contains("sensitivity")]
SEL = cand.sort_values("cv_rmse").model.iloc[0]
R["selected"] = SEL
TREE_SEL = cand[cand.model.isin(tuned.keys())].sort_values("cv_rmse").model.iloc[0]
R["selected_tree"] = TREE_SEL
print("Selected overall:", SEL, "| selected tree model:", TREE_SEL)

# Paired bootstrap of test-set differences (tree model minus comparator)
rng = np.random.default_rng(SEED)
P = {k: m.predict(Xte) for k, m in models.items()}
boots = rng.integers(0, len(yte), (2000, len(yte)))
R["boot_diff"] = {}
for comp in ["Gamma GLM", "Gamma GLM + actuarial terms", "OLS"]:
    e1, e0 = P[TREE_SEL] - yte, P[comp] - yte
    dmae = np.array([np.abs(e1[b]).mean() - np.abs(e0[b]).mean() for b in boots])
    drmse = np.array([np.sqrt((e1[b]**2).mean()) - np.sqrt((e0[b]**2).mean()) for b in boots])
    R["boot_diff"][comp] = dict(mae=float(np.abs(e1).mean() - np.abs(e0).mean()),
        mae_ci=np.percentile(dmae, [2.5, 97.5]).tolist(),
        rmse=float(rmse(yte, P[TREE_SEL]) - rmse(yte, P[comp])), rmse_ci=np.percentile(drmse, [2.5, 97.5]).tolist())
print(R["boot_diff"])

# ---------- 4. Nested repeated CV on full data (ranking stability + nested subgroup predictions) ----------
outer = group_splits(GROUPS, 5, 2, SEED + 1)
ALL = list(fixed) + list(tuned)
CACHE = f"{OUT}/nested_{HASH_CONTENT[:12]}_{BOOST.replace(' ', '_')}_{CONFIG_VERSION}_sk{sklearn.__version__}.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); nest = pd.DataFrame(list(z["nest"])); NPRED = z["npred"].item()
else:
    nest, NPRED = [], {m: np.full((2, len(y)), np.nan) for m in ALL}
    for k, (a, b) in enumerate(outer):
        rep_ = k // 5
        for name in ALL:
            if name in tuned:
                pipe, grid = tuned[name]
                m = GridSearchCV(clone(pipe), grid, cv=group_splits(GROUPS[a], seed=SEED + 100 + k),
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
            else:
                m = clone(fixed[name])
            m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b]); NPRED[name][rep_, b] = p
            nest.append(dict(fold=k, model=name, mae=mean_absolute_error(y[b], p), rmse=rmse(y[b], p), r2=r2_score(y[b], p)))
        print("outer fold", k, flush=True)
    np.savez(CACHE, nest=np.array(nest, dtype=object), npred=np.array(NPRED, dtype=object))
    nest = pd.DataFrame(nest)
ns = nest.groupby("model").agg(mae=("mae", "mean"), mae_sd=("mae", "std"), rmse=("rmse", "mean"),
                               rmse_sd=("rmse", "std"), r2=("r2", "mean"), r2_sd=("r2", "std"))
best_per_fold = nest[~nest.model.str.contains("sensitivity")].loc[
    lambda d: d.groupby("fold").rmse.idxmin()].model.value_counts().to_dict()
print(ns.round(3)); print("best by fold:", best_per_fold)
R["nested"] = ns.reset_index().to_dict("records"); R["nested_best"] = best_per_fold
R["nested_folds"] = int(nest.fold.nunique())
pv = nest.pivot(index="fold", columns="model", values="rmse")
R["nested_paired_rmse_gb_minus_rf"] = dict(mean=float((pv[BOOST] - pv["Random forest"]).mean()),
    sd=float((pv[BOOST] - pv["Random forest"]).std()), min=float((pv[BOOST] - pv["Random forest"]).min()),
    max=float((pv[BOOST] - pv["Random forest"]).max()))
nest.to_csv(f"{OUT}/nested_folds.csv", index=False)

# ---------- 5. GLM relativities and Gamma variance check ----------
def relativities(pipe_raw, Xa, ya, cols):
    pipe_raw.fit(Xa, ya); m = pipe_raw.named_steps["m"]
    names = pipe_raw.named_steps["p"].get_feature_names_out()
    return [dict(term=str(t).split("__")[-1], beta=float(b), rel=float(np.exp(b))) for t, b in zip(names, m.coef_)] + \
           [dict(term="intercept", beta=float(m.intercept_), rel=float(np.exp(m.intercept_)))]
R["glm_rel"] = relativities(Pipeline([("p", prep(False)), ("m", GammaRegressor(alpha=0.0, max_iter=20000))]), Xtr, ytr, None)
R["glm2_rel"] = relativities(Pipeline([("fe", FunctionTransformer(actuarial_terms)),
    ("p", ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT), ("num", "passthrough", NUM + EXTRA)])),
    ("m", GammaRegressor(alpha=0.0, max_iter=50000))]), Xtr, ytr, None)
# Gamma implies roughly constant coefficient of variation across fitted-value groups
fit_tr = models["Gamma GLM"].predict(Xtr)
dec = pd.qcut(fit_tr, 5, labels=False)
cvs = pd.DataFrame({"y": ytr, "d": dec}).groupby("d").y.agg(lambda s: s.std() / s.mean())
R["gamma_cv_by_quintile"] = cvs.round(3).tolist()
pear = (ytr - fit_tr) / fit_tr
R["gamma_dispersion"] = float((pear ** 2).sum() / (len(ytr) - 9))

# ---------- 5b. POST-HOC diagnostic (not eligible for selection) ----------
def smoker_terms(Z):
    Z = actuarial_terms(Z); s = (Z["smoker"] == "yes").astype(float)
    Z["smk_age"] = s * Z["age"]; return Z
PH = NUM + EXTRA + ["smk_age"]
posthoc = {
    "Post hoc: Gamma GLM + smoker interactions": Pipeline([("fe", FunctionTransformer(smoker_terms)),
        ("p", ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT), ("num", StandardScaler(), PH)])),
        ("m", GammaRegressor(alpha=0.0, max_iter=5000))]),
    "Post hoc: OLS + smoker interactions": Pipeline([("fe", FunctionTransformer(smoker_terms)),
        ("p", ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT), ("num", StandardScaler(), PH)])),
        ("m", LinearRegression())]),
}
ph_rows = []
for name, m in posthoc.items():
    cv = cross_validate(m, Xtr, ytr, cv=rkf, n_jobs=-1,
                        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    m.fit(Xtr, ytr); p = m.predict(Xte); top = yte >= np.quantile(yte, 0.9)
    ph_rows.append(dict(model=name, cv_mae=-cv["test_neg_mean_absolute_error"].mean(),
        cv_rmse=-cv["test_neg_root_mean_squared_error"].mean(), cv_r2=cv["test_r2"].mean(),
        te_mae=mean_absolute_error(yte, p), te_rmse=rmse(yte, p), te_r2=r2_score(yte, p),
        te_top10_mae=mean_absolute_error(yte[top], p[top]), te_balance=p.sum() / yte.sum()))
R["posthoc"] = ph_rows; print(pd.DataFrame(ph_rows).round(3).to_string())
ols_ph = posthoc["Post hoc: OLS + smoker interactions"]
R["posthoc_ols_coef"] = dict(zip([str(t).split("__")[-1] for t in ols_ph.named_steps["p"].get_feature_names_out()],
                                 map(float, ols_ph.named_steps["m"].coef_)))
# raw-scale additive smoker effect by age band (descriptive)
R["smoker_gap_by_age"] = df.assign(band=pd.cut(df.age, [17, 30, 45, 64])).groupby(["band", "smoker"], observed=True).charges.mean().unstack().assign(
    diff=lambda t: t["yes"] - t["no"], ratio=lambda t: t["yes"] / t["no"]).round(2).reset_index().astype({"band": str}).to_dict("records")
print(R["smoker_gap_by_age"])

# ---------- 6. Exact interventional Shapley values ----------
n = len(FEATS); SUBS = list(itertools.product([0, 1], repeat=n)); IDX = {s: i for i, s in enumerate(SUBS)}
def coalition_values(model, Xe, bg):
    Xe = Xe.reset_index(drop=True); bg = bg.reset_index(drop=True); V = np.zeros((len(Xe), len(SUBS)))
    rep_bg = pd.concat([bg] * len(Xe), ignore_index=True)
    rep_x = Xe.loc[np.repeat(np.arange(len(Xe)), len(bg))].reset_index(drop=True)
    for si, s in enumerate(SUBS):
        Z = rep_bg.copy()
        for j, f in enumerate(FEATS):
            if s[j]: Z[f] = rep_x[f].values
        V[:, si] = model.predict(Z).reshape(len(Xe), len(bg)).mean(1)
    return V
def shapley(V):
    phi = np.zeros((V.shape[0], n))
    for j in range(n):
        for s in SUBS:
            if s[j]: continue
            k = sum(s); w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            s1 = list(s); s1[j] = 1; phi[:, j] += w * (V[:, IDX[tuple(s1)]] - V[:, IDX[s]])
    return phi, V[:, IDX[(0,) * n]]
def interaction(V, i, j):  # Shapley interaction index (total, both halves)
    out = np.zeros(V.shape[0])
    for s in SUBS:
        if s[i] or s[j]: continue
        k = sum(s); w = math.factorial(k) * math.factorial(n - k - 2) / math.factorial(n - 1)
        si = list(s); si[i] = 1; sj = list(s); sj[j] = 1; sij = list(si); sij[j] = 1
        out += w * (V[:, IDX[tuple(sij)]] - V[:, IDX[tuple(si)]] - V[:, IDX[tuple(sj)]] + V[:, IDX[s]])
    return out

BG = Xtr.sample(200, random_state=SEED)
sel_model = models[TREE_SEL]
V_sel = coalition_values(sel_model, Xte, BG); phi, base = shapley(V_sel)
V_glm = coalition_values(models["Gamma GLM"], Xte, BG); phi_glm, _ = shapley(V_glm)
V_glm2 = coalition_values(models["Gamma GLM + actuarial terms"], Xte, BG); phi_glm2, _ = shapley(V_glm2)
pred_sel = sel_model.predict(Xte)
R["additivity_max_err"] = float(np.abs(base + phi.sum(1) - pred_sel).max())
R["shap_base"] = float(base.mean())
imp = pd.DataFrame({"feature": FEATS, "tree": np.abs(phi).mean(0), "glm": np.abs(phi_glm).mean(0),
                    "glm2": np.abs(phi_glm2).mean(0)}).sort_values("tree", ascending=False)
R["importance"] = imp.to_dict("records"); print(imp.round(0))
pairs = {}
for i, j in itertools.combinations(range(n), 2):
    pairs[f"{FEATS[i]} x {FEATS[j]}"] = float(np.abs(interaction(V_sel, i, j)).mean())
R["interactions"] = dict(sorted(pairs.items(), key=lambda t: -t[1])); print(list(R["interactions"].items())[:5])
Xr = Xte.reset_index(drop=True)
bp = pd.DataFrame({"smoker": Xr.smoker, "obese": Xr.bmi >= 30, "phi_bmi": phi[:, 2]})
R["bmi_phi_groups"] = bp.groupby(["smoker", "obese"]).phi_bmi.agg(["mean", "count"]).reset_index().to_dict("records")
R["raw_smk_obese"] = df.assign(obese=df.bmi >= 30).groupby(["smoker", "obese"]).charges.agg(["mean", "count"]).reset_index().to_dict("records")

# Explanation stability: 20 bootstrap refits with fixed hyperparameters, fixed explanation subset
sub = Xte.sample(100, random_state=SEED); bg_s = Xtr.sample(100, random_state=SEED + 1)
ranks, gaps, inter_sb = [], [], []
for r_ in range(20):
    bi = rng.integers(0, len(Xtr), len(Xtr))
    m = clone(sel_model).fit(Xtr.iloc[bi], ytr[bi])
    Vb = coalition_values(m, sub, bg_s); ph, _ = shapley(Vb)
    ranks.append(pd.Series(np.abs(ph).mean(0), index=FEATS).rank(ascending=False).to_dict())
    d = pd.DataFrame({"s": sub.smoker.values, "o": sub.bmi.values >= 30, "p": ph[:, 2]})
    g = d.groupby(["s", "o"]).p.mean()
    gaps.append(dict(smoker=g[("yes", True)] - g[("yes", False)], nonsmoker=g[("no", True)] - g[("no", False)]))
    inter_sb.append(np.abs(interaction(Vb, 2, 4)).mean())
ranks = pd.DataFrame(ranks); gaps = pd.DataFrame(gaps)
R["stability"] = dict(rank_mean=ranks.mean().to_dict(), rank_min=ranks.min().to_dict(), rank_max=ranks.max().to_dict(),
    top3_same=float((ranks[["smoker", "age", "bmi"]].max(1) <= 3).mean()),
    gap_smoker=[float(gaps.smoker.mean()), *np.percentile(gaps.smoker, [2.5, 97.5]).tolist()],
    gap_nonsmoker=[float(gaps.nonsmoker.mean()), *np.percentile(gaps.nonsmoker, [2.5, 97.5]).tolist()],
    inter_smk_bmi=[float(np.mean(inter_sb)), *np.percentile(inter_sb, [2.5, 97.5]).tolist()])
print(R["stability"])

# Prespecified local cases: largest positive residual, largest negative residual, median |error|
res = yte - pred_sel; ae = np.abs(res)
cases = {"Largest under-prediction": int(np.argmax(res)), "Largest over-prediction": int(np.argmin(res)),
         "Median absolute error": int(np.argsort(ae)[len(ae) // 2])}
R["local"] = {k: dict(profile={f: (v if isinstance(v, str) else float(v)) for f, v in Xr.loc[i].to_dict().items()},
                      actual=float(yte[i]), pred=float(pred_sel[i]), base=float(base[i]),
                      phi=dict(zip(FEATS, map(float, phi[i]))),
                      add_err=float(base[i] + phi[i].sum() - pred_sel[i])) for k, i in cases.items()}
for k, v in R["local"].items(): print(k, v)

# ---------- 7. Subgroup error audit from NESTED outer-fold predictions ----------
# Each record has two outer-fold predictions (2 repeats); per-record errors are averaged over the repeats.
def boot_mean(v, B=2000):
    bs = rng.integers(0, len(v), (B, len(v))); return v[bs].mean(1)
sg, sgdiff = [], []
for name in ["Gamma GLM", "Gamma GLM + actuarial terms", TREE_SEL]:
    E = NPRED[name] - y                      # signed errors, shape (2, n)
    se, ae = E.mean(0), np.abs(E).mean(0)    # per-record mean signed / absolute error
    for var in ["sex", "region", "smoker"]:
        levels = sorted(df[var].unique())
        for lvl in levels:
            idx = np.where(df[var].values == lvl)[0]; rest = np.where(df[var].values != lvl)[0]
            mb, sb = boot_mean(ae[idx]), boot_mean(se[idx])
            sg.append(dict(var=var, level=lvl, model=name, n=len(idx), obs=float(y[idx].mean()),
                           pred=float(NPRED[name][:, idx].mean()), mae=float(ae[idx].mean()),
                           mae_ci=np.percentile(mb, [2.5, 97.5]).tolist(),
                           mean_signed_error=float(se[idx].mean()), mean_signed_error_ci=np.percentile(sb, [2.5, 97.5]).tolist()))
            if var == "sex" and lvl != "female": continue
            if var == "smoker" and lvl != "yes": continue
            d_mae = boot_mean(ae[idx]) - boot_mean(ae[rest]); d_se = boot_mean(se[idx]) - boot_mean(se[rest])
            sgdiff.append(dict(var=var, level=lvl, vs=("male" if var == "sex" else "no" if var == "smoker" else "other regions"),
                               model=name, mae_diff=float(ae[idx].mean() - ae[rest].mean()),
                               mae_diff_ci=np.percentile(d_mae, [2.5, 97.5]).tolist(),
                               se_diff=float(se[idx].mean() - se[rest].mean()), se_diff_ci=np.percentile(d_se, [2.5, 97.5]).tolist()))
R["subgroup"] = sg; R["subgroup_diff"] = sgdiff
print(pd.DataFrame(sg)[["var", "level", "model", "n", "mae", "mean_signed_error"]].round(0).to_string())
print(pd.DataFrame(sgdiff).round(0).to_string())

# ---------- 8. Figures ----------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
from matplotlib.ticker import FuncFormatter
KF = FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0")
NAVY, RED, GREY, BLUE = "#1f3b5c", "#b03a2e", "#8c8c8c", "#4a7fb5"
show = ["Gamma GLM", "Gamma GLM + actuarial terms", TREE_SEL]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
for c, name in enumerate(show):
    p = P[name]; col = np.where(Xte.smoker.values == "yes", RED, NAVY)
    ax[0, c].scatter(yte, p, s=5, c=col, alpha=0.7); lim = [0, max(yte.max(), p.max()) * 1.03]
    ax[0, c].plot(lim, lim, "k--", lw=0.7); ax[0, c].set_title(name, fontsize=8)
    ax[0, c].set_xlabel("Observed charges (USD)"); ax[1, c].scatter(p, yte - p, s=5, c=col, alpha=0.7)
    ax[1, c].axhline(0, color="k", lw=0.7); ax[1, c].set_xlabel("Predicted charges (USD)")
for a_ in ax.flat: a_.xaxis.set_major_formatter(KF); a_.yaxis.set_major_formatter(KF)
ax[0, 0].set_ylabel("Predicted (USD)"); ax[1, 0].set_ylabel("Residual: observed − predicted")
ax[0, 2].scatter([], [], c=RED, s=8, label="smoker"); ax[0, 2].scatter([], [], c=NAVY, s=8, label="non-smoker")
ax[0, 2].legend(frameon=False, fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_fit.png"); plt.close()

order = imp.feature.tolist()
fig, ax = plt.subplots(figsize=(6.6, 3.0))
jit = np.random.default_rng(1)
BIN = {"smoker": ("yes", "no"), "sex": ("male", "female")}
REG = {"northeast": "#4c72b0", "northwest": "#55a868", "southeast": "#c44e52", "southwest": "#8172b2"}
from matplotlib.lines import Line2D
for r_, f in enumerate(order[::-1]):
    yy = r_ + jit.uniform(-0.28, 0.28, len(Xr)); xx = phi[:, FEATS.index(f)]
    if f in NUM:
        v = Xr[f]; v = (v - v.min()) / (v.max() - v.min())
        sc = ax.scatter(xx, yy, c=v, cmap="coolwarm", s=5, vmin=0, vmax=1)
    elif f in BIN:
        a1, a0 = BIN[f]; col = np.where(Xr[f].values == a1, "#d17a00", "#5a5a5a")
        ax.scatter(xx, yy, c=col, s=5)
    else:
        ax.scatter(xx, yy, c=[REG[v] for v in Xr[f]], s=5)
ax.set_yticks(range(n)); ax.set_yticklabels(order[::-1]); ax.axvline(0, color="k", lw=0.6)
ax.xaxis.set_major_formatter(KF); ax.set_xlabel(f"Shapley value (USD; contribution to {TREE_SEL} prediction)")
cb = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.04); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
cb.set_label("age, BMI, children", fontsize=6.5); cb.ax.tick_params(labelsize=6.5)
h = [Line2D([], [], marker="o", ls="", color="#d17a00", ms=4, label="smoker = yes / sex = male"),
     Line2D([], [], marker="o", ls="", color="#5a5a5a", ms=4, label="smoker = no / sex = female")] + \
    [Line2D([], [], marker="o", ls="", color=c, ms=4, label=k) for k, c in REG.items()]
ax.legend(handles=h, frameon=False, fontsize=6, loc="lower right", ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_beeswarm.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, (ph_, ttl) in zip(ax, [(phi, TREE_SEL), (phi_glm, "Gamma GLM"), (phi_glm2, "Gamma GLM + actuarial terms")]):
    for s_, c in [("no", NAVY), ("yes", RED)]:
        m_ = Xr.smoker == s_
        a.scatter(Xr.bmi[m_], ph_[m_, 2], s=6, c=c, alpha=0.75, label=f"smoker = {s_}")
    a.axvline(30, ls=":", color="k", lw=0.7); a.set_title(ttl, fontsize=8); a.set_xlabel("BMI")
ax[0].yaxis.set_major_formatter(KF); ax[0].set_ylabel("Shapley value of BMI (USD)"); ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_bmi_dependence.png"); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
for a, (lab, L) in zip(axs, R["local"].items()):
    ph_ = pd.Series(L["phi"])[FEATS[::-1]]
    a.barh(FEATS[::-1], ph_, color=[RED if v > 0 else NAVY for v in ph_]); a.axvline(0, color="k", lw=0.6)
    a.set_title(f"{lab}\npredicted \\${L['pred']:,.0f}; observed \\${L['actual']:,.0f}", fontsize=7)
    a.tick_params(labelsize=7)
[a_.xaxis.set_major_formatter(KF) for a_ in axs]; axs[1].set_xlabel("Contribution relative to baseline prediction (USD)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_local.png"); plt.close()

json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done")
