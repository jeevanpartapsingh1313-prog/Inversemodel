import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path                    # <— needed for Path()
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold                         # <— needed for StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Try to use the labeled file first
if Path("labeled_cilia_video_dataset.csv").exists():
    df = pd.read_csv("labeled_cilia_video_dataset.csv")
    print("Loaded labeled_cilia_video_dataset.csv")
elif Path("cbf_per_video.csv").exists():
    df = pd.read_csv("cbf_per_video.csv")
    print("Loaded cbf_per_video.csv")

    # if class column doesn't exist, create it from label
    if "class" not in df.columns:
        if "label" not in df.columns:
            raise RuntimeError(
                "cbf_per_video.csv has no 'label' or 'class' column. "
                "Run your feature-extraction script that adds labels first."
            )
        df["class"] = df["label"].apply(
            lambda x: "Healthy" if "Healthy" in x else "PCD"
        )
else:
    raise FileNotFoundError(
        "No labeled_cilia_video_dataset.csv or cbf_per_video.csv in this folder."
    )

print("\nFirst few rows:")
print(df.head())
print("\nClass counts:")
print(df["class"].value_counts())

# -------------------- 2. PICK FEATURES --------------------

feature_cols = [
    "global_cbf_hz",
    "global_peak_amp",
    "global_var",
    "global_zcr",
    "cbf_mean_tiles",
    "cbf_std_tiles",
    "cbf_min_tiles",
    "cbf_max_tiles",
    "frac_low_cbf",
    "frac_normal_cbf",
    "frac_high_cbf",
]

# Keep only rows where all features + class are not NaN
df_clean = df.dropna(subset=feature_cols + ["class"])

X = df_clean[feature_cols]
y = df_clean["class"]

print("\nUsing", X.shape[0], "videos with complete feature data.")

# -------------------- 3. TRAIN / TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,      # 30% for testing
    random_state=42,
    stratify=y          # keep class balance
)

print("\nTrain size:", X_train.shape[0], "videos")
print("Test size:", X_test.shape[0], "videos")

# -------------------- 4. TRAIN RANDOM FOREST --------------------

rf = RandomForestClassifier(
    n_estimators=300,   # number of trees
    max_depth=None,     # let trees grow fully
    random_state=42,
    class_weight="balanced"  # helps if classes are imbalanced
)

rf.fit(X_train, y_train)

# -------------------- 5. EVALUATE --------------------

y_pred = rf.predict(X_test)

print("\n=== Test Performance ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------- 6. CROSS-VALIDATION (OPTIONAL, FOR STABILITY) --------------------

min_per_class = y.value_counts().min()
n_splits = max(2, min(3, min_per_class))  # between 2 and 3 folds

print(f"\nUsing {n_splits}-fold cross-validation (limited by smallest class size={min_per_class})")

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv)

print(f"\n{n_splits}-fold cross-validation accuracy scores:", scores)
print("Mean CV accuracy:", round(scores.mean(), 3))

# -------------------- 7. FEATURE IMPORTANCE PLOT --------------------

importances = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feat_names, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
