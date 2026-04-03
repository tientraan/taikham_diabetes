import os
import shutil
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from lightgbm import LGBMClassifier

st.set_page_config(page_title="LightGBM tối ưu", layout="wide")
st.title("🚀 Phân loại tái nhập viện (LightGBM tối ưu)")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_best.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/diabetic_data.csv")


def prepare_data(df):
    data = df.copy()
    data = data.replace("?", np.nan)

    # 1 = có tái nhập viện, 0 = không tái nhập viện
    data["target"] = data["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

    # Bỏ cột không cần và cột dễ gây leakage
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "weight",
        "payer_code",
        "medical_specialty",
        "discharge_disposition_id"
    ]
    drop_cols = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=drop_cols)

    # Chọn feature mạnh hơn để giảm nhiễu
    selected_features = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed"
    ]

    selected_features = [c for c in selected_features if c in data.columns]

    X = data[selected_features]
    y = data["target"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, cat_cols, num_cols


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = 0.0

    for th in np.arange(0.30, 0.71, 0.01):
        y_pred_temp = (y_prob >= th).astype(int)
        f1_temp = f1_score(y_true, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = float(th)

    return best_threshold, best_f1


@st.cache_resource
def train_or_load_model(df):
    X, y, cat_cols, num_cols = prepare_data(df)

    # Chia 3 tập: train / val / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return (
            saved["pipeline"],
            X_test,
            y_test,
            X,
            saved["best_threshold"],
            saved["best_params"],
            saved["best_cv_f1"]
        )

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__learning_rate": [0.03, 0.05, 0.07],
        "model__num_leaves": [31, 63, 127],
        "model__max_depth": [-1, 6, 8, 10],
        "model__min_child_samples": [10, 20, 30],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.1, 0.2, 0.5],
        "model__reg_lambda": [1, 2, 3, 5]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val, y_val_prob)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "best_threshold": best_threshold,
            "best_params": search.best_params_,
            "best_cv_f1": search.best_score_,
            "best_val_f1": best_val_f1
        },
        MODEL_PATH
    )

    return best_pipeline, X_test, y_test, X, best_threshold, search.best_params_, search.best_score_


def reset_model():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    st.cache_resource.clear()
    st.success("Đã xóa model cũ. Hãy tải lại trang để train lại.")


df = load_data()
pipeline, X_test, y_test, X_full, best_threshold, best_params, best_cv_f1 = train_or_load_model(df)

page = st.sidebar.radio("Chọn trang", [
    "Trang 1: EDA",
    "Trang 2: Dự đoán",
    "Trang 3: Đánh giá"
])

st.sidebar.markdown("---")
if st.sidebar.button("Xóa model cũ và train lại"):
    reset_model()

if page == "Trang 1: EDA":
    st.header("📊 Khám phá dữ liệu")

    st.subheader("5 dòng đầu")
    st.dataframe(df.head())

    st.subheader("Kích thước dữ liệu")
    st.write(f"Số dòng: {df.shape[0]}")
    st.write(f"Số cột: {df.shape[1]}")

    st.subheader("Phân bố readmitted")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["readmitted"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Phân bố readmitted")
    ax.set_xlabel("Nhóm")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)

    st.subheader("Tỷ lệ mục tiêu sau khi đổi nhãn")
    target_counts = pd.Series(df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)).value_counts()
    st.write("0 = Không tái nhập viện, 1 = Có tái nhập viện")
    st.write(target_counts)

elif page == "Trang 2: Dự đoán":
    st.header("🔮 Dự đoán")

    X_raw, _, _, _ = prepare_data(df)

    input_data = {}
    col1, col2 = st.columns(2)
    columns = X_raw.columns.tolist()

    for i, col in enumerate(columns):
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if X_raw[col].dtype == "object":
                options = X_raw[col].dropna().astype(str).unique().tolist()
                options = sorted(options)
                input_data[col] = st.selectbox(col, options)
            else:
                input_data[col] = st.number_input(
                    col,
                    min_value=float(X_raw[col].min()),
                    max_value=float(X_raw[col].max()),
                    value=float(X_raw[col].median())
                )

    input_df = pd.DataFrame([input_data])

    st.write(f"Ngưỡng tối ưu theo F1: **{best_threshold:.2f}**")

    if st.button("Dự đoán"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= best_threshold else 0

        st.write(f"Xác suất tái nhập viện: **{prob:.4f}**")

        if pred == 1:
            st.error("Kết luận: Có nguy cơ tái nhập viện")
        else:
            st.success("Kết luận: Nguy cơ thấp")

else:
    st.header("📈 Đánh giá LightGBM")

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    st.write(f"Best threshold tìm được: **{best_threshold:.2f}**")
    st.write(f"Best CV F1-score: **{best_cv_f1:.4f}**")

    with st.expander("Xem tham số tốt nhất"):
        st.json(best_params)

    use_auto = st.checkbox("Dùng threshold tối ưu", value=True)

    if use_auto:
        threshold = best_threshold
    else:
        threshold = st.slider("Threshold", 0.10, 0.90, 0.45, 0.01)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{pre:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    st.pyplot(fig)

    st.subheader("Báo cáo phân loại")
    st.text(classification_report(y_test, y_pred, zero_division=0))