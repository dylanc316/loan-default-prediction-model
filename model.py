import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

ZIP_PATH = r"C:\Users\dylan\mortgage_projects\loan_default_predictor\Performance_All.zip"
SAMPLE_MOD = 250
RANDOM_STATE = 42

with zipfile.ZipFile(ZIP_PATH) as z:
    members = {name.split("/")[-1]: name for name in z.namelist()}
    files = sorted([
        name for name in members
        if name.endswith(".csv") and len(name) == 10
    ])[-20:]

    rows = []

    for filename in files:
        print("Reading", filename)

        with z.open(members[filename]) as f:
            for raw in f:
                line = raw.decode("utf-8").rstrip("\r\n")
                first_fields = line.split("|", 2)

                if len(first_fields) < 3 or not first_fields[1].isdigit():
                    continue

                if int(first_fields[1][-6:]) % SAMPLE_MOD != 0:
                    continue

                fields = line.split("|")

                if len(fields) < 41:
                    continue

                rows.append([
                    fields[1], fields[2], fields[3],
                    fields[7], fields[8], fields[9], fields[11], fields[12],
                    fields[13], fields[19], fields[20], fields[22], fields[23],
                    fields[26], fields[27], fields[29], fields[30], fields[31],
                    fields[34], fields[39], fields[40]
                ])

data = pd.DataFrame(rows, columns=[
    "loan_id", "date", "channel",
    "orig_rate", "current_rate", "current_upb", "orig_upb", "orig_term",
    "orig_date", "orig_ltv", "orig_cltv", "dti", "fico",
    "purpose", "property_type", "occupancy", "state", "msa",
    "amortization_type", "dq", "history"
])

data["date"] = pd.to_datetime(data["date"], format="%m%Y", errors="coerce")
data["orig_date"] = pd.to_datetime(data["orig_date"], format="%m%Y", errors="coerce")

data = (
    data.dropna(subset=["loan_id", "date"])
    .drop_duplicates(["loan_id", "date"])
    .sort_values(["loan_id", "date"])
    .reset_index(drop=True)
)

for column in [
    "orig_rate", "current_rate", "current_upb", "orig_upb", "orig_term",
    "orig_ltv", "orig_cltv", "dti", "fico", "dq"
]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["dq"] = data["dq"].fillna(0)

data["age_months"] = (
    (data["date"].dt.year - data["orig_date"].dt.year) * 12
    + data["date"].dt.month
    - data["orig_date"].dt.month
)

data["rate_change"] = data["current_rate"] - data["orig_rate"]
data["balance_ratio"] = data["current_upb"] / data["orig_upb"]

history = data["history"].fillna("").astype(str)
history_columns = []

for i in range(24):
    column = "hist_" + str(i + 1)
    history_columns.append(column)
    data[column] = pd.to_numeric(
        history.str.slice(i * 2, i * 2 + 2),
        errors="coerce"
    )

data["hist_max_24m"] = data[history_columns].max(axis=1)
data["hist_max_12m"] = data[history_columns[-12:]].max(axis=1)
data["hist_max_6m"] = data[history_columns[-6:]].max(axis=1)
data["hist_months_30plus"] = (data[history_columns] >= 1).sum(axis=1)
data["hist_months_60plus"] = (data[history_columns] >= 2).sum(axis=1)
data["hist_months_90plus"] = (data[history_columns] >= 3).sum(axis=1)

data["bad_date"] = data["date"].where(data["dq"] >= 3)

data["next_bad_date"] = (
    data["bad_date"].iloc[::-1]
    .groupby(data["loan_id"].iloc[::-1], sort=False)
    .ffill()
    .iloc[::-1]
)

last_date = data["date"].max()

data = data[
    (data["dq"] < 3) &
    (data["date"] <= last_date - pd.DateOffset(months=12))
].copy()

data["target"] = (
    data["next_bad_date"].notna() &
    (data["next_bad_date"] <= data["date"] + pd.DateOffset(months=12))
).astype(int)

numeric_features = [
    "orig_rate",
    "current_rate",
    "current_upb",
    "orig_upb",
    "orig_term",
    "orig_ltv",
    "orig_cltv",
    "dti",
    "fico",
    "dq",
    "age_months",
    "rate_change",
    "balance_ratio",
    "hist_max_24m",
    "hist_max_12m",
    "hist_max_6m",
    "hist_months_30plus",
    "hist_months_60plus",
    "hist_months_90plus"]
# ] + history_columns

# categorical_features = [
    # "channel",
    # "purpose",
    # "property_type",
    # "occupancy",
    # "state",
    # "msa",
#     "amortization_type"
# ]

# X = pd.concat(
#     [
#         data[numeric_features],
#         pd.get_dummies(data[categorical_features].fillna("MISSING"), dtype=float)
#     ],
#     axis=1
# )

X = data[numeric_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

cutoff = data["date"].quantile(0.75)
train_end = cutoff - pd.DateOffset(months=12)

train = data["date"] < train_end
test = data["date"] >= cutoff

X_train = X.loc[train].copy()
X_test = X.loc[test].copy()
y_train = data.loc[train, "target"]
y_test = data.loc[test, "target"]

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("\nSampled observations:", len(data))
print("Train observations:", len(X_train))
print("Test observations:", len(X_test))
print("Test default rate:", round(y_test.mean(), 5))
print("")



#LOGISTIC REGRESSION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight=None,
    random_state=RANDOM_STATE
)

logistic_model.fit(X_train_scaled, y_train)

logistic_probability = logistic_model.predict_proba(X_test_scaled)[:, 1]
logistic_auc = roc_auc_score(y_test, logistic_probability)
logistic_ar = 2 * logistic_auc - 1

print("Logistic regression AUC:", round(logistic_auc, 4))
print("Logistic regression Accuracy Ratio:", round(logistic_ar, 4))

logistic_fpr, logistic_tpr, logistic_thresholds = roc_curve(y_test, logistic_probability)

ks_values = logistic_tpr - logistic_fpr
ks = np.max(ks_values)
ks_index = np.argmax(ks_values)
ks_threshold = logistic_thresholds[ks_index]

print("Logistic regression KS Statistic:", round(ks, 4))
print("Logistic regression KS Threshold:", round(ks_threshold, 4))
print("")



#XGBOOST
xgb_model = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=10,
    reg_alpha=0.2,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_probability = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probability)
xgb_ar = 2 * xgb_auc - 1

print("XGBoost AUC:", round(xgb_auc, 4))
print("XGBoost Accuracy Ratio:", round(xgb_ar, 4))

xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_probability)

ks_values = xgb_tpr - xgb_fpr
ks = np.max(ks_values)
ks_index = np.argmax(ks_values)
ks_threshold = xgb_thresholds[ks_index]

print("XGBoost KS Statistic:", round(ks, 4))
print("XGBoost KS Threshold:", round(ks_threshold, 4))
print("")

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    color="darkgreen",
    linewidth=2,
    label=f"Logistic regression (AUC = {logistic_auc:.4f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    color="darkblue",
    linewidth=2,
    label=f"XGBoost (AUC = {xgb_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Random classifier (AUC = 0.50)"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for 12-Month Mortgage Default Prediction")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.grid(alpha=0.25)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

logistic_importance = (
    pd.Series(logistic_model.coef_[0], index=X_train.columns)
    .sort_values(key=abs, ascending=False)
    .head(20)
)

print("\nTop 20 feature importances for Logistic regression:")
print(logistic_importance.to_string())
print("")

xgb_importance = (
    pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
    .head(20)
)

print("\nTop 20 feature importances for XGBoost:")
print(xgb_importance.to_string())
