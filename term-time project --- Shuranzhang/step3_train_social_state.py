from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_with_features.csv"


def main():

    # 1) Load + clean
    df = pd.read_csv(DATA)

    # only keep images where a face was detected
    if "has_face" in df.columns:
        df = df[df["has_face"] == 1].copy()

    # clean target labels
    if "social_state" not in df.columns:
        raise ValueError("Column 'social_state' not found in dataset_with_features.csv")

    df["social_state"] = df["social_state"].astype(str).str.strip().str.lower()
    df = df[df["social_state"].isin(["interactive", "reactive", "isolated"])].copy()

    # features
    cat_cols = ["context", "eyes", "mouth", "eyebrow", "emotion"]
    num_cols = ["intensity", "eye_open", "mouth_open", "mouth_width", "smile", "brow_raise", "activity"]

    # sanity checks (helpful if your CSV missing something)
    missing_cols = [c for c in (cat_cols + num_cols) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    X = df[cat_cols + num_cols].copy()
    y = df["social_state"].copy()

    print("Samples (face detected & valid label):", len(df))
    print("Class counts:\n", y.value_counts(), "\n")


    # 2) Preprocess + model
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
        ]
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    model = Pipeline(steps=[("pre", pre), ("clf", clf)])


    # 3) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,  # IMPORTANT for small dataset
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)


    # 4) Report
    labels = ["interactive", "isolated", "reactive"]  # fixed order for plots

    print("\n Classification Report (social_state)")
    print(classification_report(y_test, pred, labels=labels, digits=3))

    cm = confusion_matrix(y_test, pred, labels=labels)
    print("\n Confusion Matrix (counts; label order fixed)")
    print("labels =", labels)
    print(cm)


    # 5) Plot: Confusion Matrix (row-normalized + count)
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cmn)

    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (social_state) — row-normalized")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cmn[i, j]:.2f})",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = ROOT / "confusion_matrix_social_state.png"
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)
    print("Saved:", cm_path)


    # 6) Plot: Feature Importance (LogReg coefficients)
    pre_fitted = model.named_steps["pre"]
    clf_fitted = model.named_steps["clf"]

    # Get expanded feature names: one-hot cat + numeric columns
    cat_ohe = pre_fitted.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_ohe.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([cat_feature_names, np.array(num_cols, dtype=object)])

    coef = clf_fitted.coef_  # shape: (n_classes, n_features)
    if coef.shape[1] != feature_names.shape[0]:
        raise ValueError(
            f"Feature name length mismatch: coef has {coef.shape[1]} features "
            f"but feature_names has {feature_names.shape[0]}."
        )

    # Global importance: mean absolute coefficient across classes
    importance = np.mean(np.abs(coef), axis=0)

    top_n = 15
    idx = np.argsort(importance)[-top_n:]
    top_features = feature_names[idx]
    top_scores = importance[idx]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(range(top_n), top_scores)
    ax.set_yticks(range(top_n), top_features)
    ax.set_xlabel("Mean |coefficient| across classes")
    ax.set_title(f"Top {top_n} Feature Importance — Logistic Regression")
    plt.tight_layout()

    fi_path = ROOT / "feature_importance_top15.png"
    plt.savefig(fi_path, dpi=300)
    plt.close(fig)
    print("Saved:", fi_path)

    print("\nDone.")


if __name__ == "__main__":
    main()