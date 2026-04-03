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

st.set_page_config(page_title="Phân loại tái nhập viện", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #2c3e50;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.block-container {
    padding-top: 2rem;
}
.stButton > button {
    background-color: #3498db;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    border: none;
}
.stButton > button:hover {
    background-color: #2980b9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CẤU HÌNH CHUNG
# =========================
TITLE = "Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường"
STUDENT_NAME = "Họ tên sinh viên: Trần Quang Tiến"
STUDENT_ID = "MSSV: 22T1020760"
TOPIC_DESC = (
    "Ứng dụng mô hình học máy để phân loại bệnh nhân có nguy cơ tái nhập viện "
    "dựa trên dữ liệu bệnh án, từ đó hỗ trợ đánh giá mức độ rủi ro và giúp "
    "ưu tiên theo dõi các trường hợp cần chú ý."
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_best.pkl")


# =========================
# HÀM TẢI DỮ LIỆU
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/diabetic_data.csv")


# =========================
# TIỀN XỬ LÝ
# =========================
def prepare_data(df):
    data = df.copy()
    data = data.replace("?", np.nan)

    data["target"] = data["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

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

    X = data[selected_features].copy()
    y = data["target"].copy()

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, cat_cols, num_cols, data


# =========================
# TÌM THRESHOLD TỐI ƯU
# =========================
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


# =========================
# TRAIN / LOAD MODEL
# =========================
@st.cache_resource
def train_or_load_model(df):
    X, y, cat_cols, num_cols, _ = prepare_data(df)

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
        required_keys = ["pipeline", "best_threshold", "best_params", "best_cv_f1", "feature_cols"]
        if all(k in saved for k in required_keys):
            return (
                saved["pipeline"],
                X_test,
                y_test,
                X,
                saved["best_threshold"],
                saved["best_params"],
                saved["best_cv_f1"],
                saved["feature_cols"]
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
            n_jobs=1,
            verbose=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [300, 500, 800],
        "model__learning_rate": [0.03, 0.05, 0.07],
        "model__num_leaves": [31, 63, 127],
        "model__max_depth": [-1, 6, 8, 10],
        "model__min_child_samples": [10, 20, 30],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.1, 0.2],
        "model__reg_lambda": [1, 2, 3]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=15,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=1,
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
            "best_val_f1": best_val_f1,
            "feature_cols": X.columns.tolist()
        },
        MODEL_PATH
    )

    return best_pipeline, X_test, y_test, X, best_threshold, search.best_params_, search.best_score_, X.columns.tolist()


def reset_model():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    st.cache_resource.clear()
    st.success("Đã xóa model cũ. Tải lại trang để train lại.")


# =========================
# LOAD
# =========================
df = load_data()
pipeline, X_test, y_test, X_full, best_threshold, best_params, best_cv_f1, feature_cols = train_or_load_model(df)

st.markdown("""
<h1 style='text-align: center; color: #2c3e50;'>
🚀 ỨNG DỤNG MACHINE LEARNING
</h1>
<h3 style='text-align: center; color: gray;'>
Phân loại nguy cơ tái nhập viện bệnh nhân tiểu đường
</h3>
<hr>
""", unsafe_allow_html=True)

st.subheader(TITLE)

page = st.sidebar.radio(
    "Chọn trang",
    [
        "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)",
        "Trang 2: Triển khai mô hình",
        "Trang 3: Đánh giá & Hiệu năng"
    ]
)

st.sidebar.markdown("---")
if st.sidebar.button("Xóa model cũ và train lại"):
    reset_model()


# =========================
# TRANG 1
# =========================
if page == "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)":
    st.header("📘 Giới thiệu đề tài")
    st.write(f"**Tên đề tài:** Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường dựa trên dữ liệu bệnh án bằng Random Forest nhằm hỗ trợ phát hiện sớm bệnh nhân có rủi ro cao")
    st.write(f"**{STUDENT_NAME}**")
    st.write(f"**{STUDENT_ID}**")
    st.write(f"**Mô tả ngắn gọn giá trị thực tiễn:** {TOPIC_DESC}")

    st.header("📊 Khám phá dữ liệu (EDA)")

    st.markdown("### 📋 Dữ liệu mẫu")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Kích thước dữ liệu")
    c1, c2 = st.columns(2)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])

    st.subheader("Biểu đồ 1: Phân bố nhãn readmitted")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    df["readmitted"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Phân bố biến readmitted")
    ax1.set_xlabel("Nhóm")
    ax1.set_ylabel("Số lượng")
    st.pyplot(fig1)

    st.subheader("Biểu đồ 2: Phân bố biến mục tiêu sau khi đổi nhãn")
    target_series = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    target_series.value_counts().sort_index().plot(kind="bar", ax=ax2)
    ax2.set_title("Phân bố target (0 = Không tái nhập viện, 1 = Có tái nhập viện)")
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Số lượng")
    st.pyplot(fig2)

    st.subheader("Nhận xét dữ liệu")
    st.write("""
    Dữ liệu gồm nhiều đặc trưng liên quan đến hồ sơ bệnh án, thuốc điều trị và lịch sử thăm khám.
    Sau khi chuyển đổi nhãn, bài toán trở thành phân loại nhị phân: có tái nhập viện và không tái nhập viện.
    Dữ liệu có cả biến số và biến phân loại, do đó cần tiền xử lý bằng cách điền giá trị thiếu và mã hóa One-Hot Encoding.
    Một số đặc trưng như số ngày nằm viện, số thuốc, số lần nhập viện trước đó và kết quả xét nghiệm được xem là có ảnh hưởng
    đáng kể đến nguy cơ tái nhập viện.
    """)


# =========================
# TRANG 2
# =========================
elif page == "Trang 2: Triển khai mô hình":
    st.header("🛠️ Triển khai mô hình")

    st.write("Người dùng nhập thông tin bệnh nhân để hệ thống dự đoán nguy cơ tái nhập viện.")
    st.write(f"**Ngưỡng dự đoán đang dùng:** {best_threshold:.2f}")

    X_raw, _, cat_cols, num_cols, _ = prepare_data(df)
    X_raw = X_raw[feature_cols].copy()

    st.subheader("Thiết kế giao diện nhập liệu")
    input_data = {}
    col1, col2 = st.columns(2)
    columns = X_raw.columns.tolist()

    for i, col in enumerate(columns):
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if col in cat_cols:
                options = sorted(X_raw[col].dropna().astype(str).unique().tolist())

                if len(options) == 0:
                    input_data[col] = ""
                    st.text_input(col, value="", disabled=True)
                else:
                    default_index = 0
                    if "None" in options:
                        default_index = options.index("None")
                    input_data[col] = st.selectbox(col, options, index=default_index)
            else:
                series = pd.to_numeric(X_raw[col], errors="coerce").dropna()

                if series.empty:
                    input_data[col] = 0.0
                    st.number_input(col, value=0.0, disabled=False)
                else:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    median_val = float(series.median())

                    if pd.api.types.is_integer_dtype(X_raw[col]):
                        input_data[col] = st.number_input(
                            col,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(median_val),
                            step=1
                        )
                    else:
                        input_data[col] = st.number_input(
                            col,
                            min_value=min_val,
                            max_value=max_val,
                            value=median_val
                        )

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols)

    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    for col in num_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    st.subheader("Xử lý logic")
    st.write("Dữ liệu đầu vào sẽ được tiền xử lý giống như lúc huấn luyện mô hình, sau đó đưa vào LightGBM để dự đoán.")
    st.dataframe(input_df, width="stretch")

    if st.button("Dự đoán"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= best_threshold else 0

        st.subheader("Kết quả dự đoán")
        if pred == 1:
            st.error("Kết luận: Bệnh nhân **có nguy cơ tái nhập viện**.")
        else:
            st.success("Kết luận: Bệnh nhân **có nguy cơ thấp**.")

        st.info(f"Độ tin cậy / Xác suất dự đoán: **{prob:.4f}**")


# =========================
# TRANG 3
# =========================
else:
    st.header("📈 Đánh giá & Hiệu năng")

    y_prob = pipeline.predict_proba(X_test)[:, 1]

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

    st.subheader("Các chỉ số đo lường")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{pre:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("Biểu đồ kỹ thuật: Confusion Matrix")
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

    st.subheader("Phân tích sai số và hướng cải thiện")
    st.write(
        f"""
        Mô hình hiện tại đạt Accuracy = {acc:.4f}, Precision = {pre:.4f}, Recall = {rec:.4f}, F1-score = {f1:.4f}.
        Kết quả cho thấy mô hình có khả năng nhận diện khá tốt các trường hợp tái nhập viện, đặc biệt Recall tương đối cao,
        giúp giảm nguy cơ bỏ sót bệnh nhân cần theo dõi. Tuy nhiên Accuracy chưa quá cao do mô hình vẫn có một số dự đoán nhầm
        giữa hai lớp. Trong tương lai, có thể cải thiện bằng cách bổ sung đặc trưng quan trọng hơn, thử thêm các thuật toán
        khác như XGBoost/CatBoost hoặc tối ưu sâu hơn threshold và siêu tham số.
        """
    )

    with st.expander("Xem tham số tốt nhất của mô hình"):
        st.json(best_params)

    st.write(f"Best CV F1-score trong quá trình tìm tham số: **{best_cv_f1:.4f}**")