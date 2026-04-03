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

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>
    .main {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #0f172a;
        letter-spacing: -0.3px;
    }

    .app-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 1rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .note-blue {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 8px;
        line-height: 1.6;
    }

    .note-green {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 8px;
        line-height: 1.6;
    }

    .note-orange {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 8px;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 10px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        justify-content: center;
    }

    div[data-testid="stMetricValue"] {
        justify-content: center;
    }

    .stButton > button {
        width: 100%;
        height: 44px;
        border-radius: 12px;
        border: none;
        background: #2563eb;
        color: white;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 10px;
    }

    .sidebar-note {
        font-size: 14px;
        color: #64748b;
        line-height: 1.5;
    }

    .small-muted {
        color: #64748b;
        font-size: 14px;
    }

    hr {
        margin: 0.7rem 0 1rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CẤU HÌNH CHUNG
# =========================
TITLE = "Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường"
STUDENT_NAME = "Họ tên sinh viên: Nguyễn Văn A"
STUDENT_ID = "MSSV: 22T1020XXX"
TOPIC_DESC = (
    "Ứng dụng mô hình học máy để phân loại bệnh nhân có nguy cơ tái nhập viện "
    "dựa trên dữ liệu bệnh án, từ đó hỗ trợ đánh giá mức độ rủi ro và giúp ưu tiên "
    "theo dõi các trường hợp cần chú ý."
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

    X = data[selected_features]
    y = data["target"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, cat_cols, num_cols, data


# =========================
# TÌM THRESHOLD
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
        if all(k in saved for k in ["pipeline", "best_threshold", "best_params", "best_cv_f1"]):
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
    st.success("Đã xóa model cũ. Tải lại trang để train lại.")


# =========================
# LOAD
# =========================
df = load_data()
pipeline, X_test, y_test, X_full, best_threshold, best_params, best_cv_f1 = train_or_load_model(df)

# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="app-title">🏥 Ứng dụng phân loại tái nhập viện</div>
<div class="app-subtitle">{TITLE}</div>
<hr>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown('<div class="sidebar-title">📂 Menu</div>', unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "Trang 1: Giới thiệu & EDA",
    "Trang 2: Dự đoán",
    "Trang 3: Đánh giá"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-note">Bấm nút bên dưới nếu muốn xóa mô hình cũ và huấn luyện lại theo code mới.</div>',
    unsafe_allow_html=True
)

if st.sidebar.button("🗑️ Xóa model cũ và train lại"):
    reset_model()

# =========================
# TRANG 1
# =========================
if page == "Trang 1: Giới thiệu & EDA":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📘 Giới thiệu đề tài")
    st.write(f"**Tên đề tài:** {TITLE}")
    st.write(f"**{STUDENT_NAME}**")
    st.write(f"**{STUDENT_ID}**")
    st.markdown(
        f'<div class="note-blue"><b>Mô tả giá trị thực tiễn:</b> {TOPIC_DESC}</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Khám phá dữ liệu")
    st.markdown("### 📋 Một phần dữ liệu thô")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Biểu đồ 1: Phân bố nhãn readmitted")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df["readmitted"].value_counts().plot(kind="bar", ax=ax1, color="#3b82f6")
        ax1.set_title("Phân bố biến readmitted")
        ax1.set_xlabel("Nhóm")
        ax1.set_ylabel("Số lượng")
        for spine in ax1.spines.values():
            spine.set_visible(False)
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Biểu đồ 2: Phân bố biến mục tiêu")
        target_series = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        target_series.value_counts().sort_index().plot(kind="bar", ax=ax2, color="#10b981")
        ax2.set_title("Target (0 = Không tái nhập viện, 1 = Có tái nhập viện)")
        ax2.set_xlabel("Target")
        ax2.set_ylabel("Số lượng")
        for spine in ax2.spines.values():
            spine.set_visible(False)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Giải thích và nhận xét")
    st.markdown(
        """
        <div class="note-green">
        Dữ liệu gồm nhiều đặc trưng liên quan đến hồ sơ bệnh án, thuốc điều trị và lịch sử thăm khám.
        Sau khi đổi nhãn, bài toán trở thành phân loại nhị phân: có tái nhập viện và không tái nhập viện.
        Dữ liệu có cả biến số và biến phân loại nên cần điền giá trị thiếu và mã hóa One-Hot Encoding.
        Một số đặc trưng như thời gian nằm viện, số thuốc, số lần nhập viện trước đó và chỉ số xét nghiệm
        được xem là có ảnh hưởng đáng kể đến nguy cơ tái nhập viện.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TRANG 2
# =========================
elif page == "Trang 2: Dự đoán":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🛠️ Triển khai mô hình")
    st.markdown(
        f'<div class="note-blue">Ngưỡng dự đoán hiện tại: <b>{best_threshold:.2f}</b></div>',
        unsafe_allow_html=True
    )
    st.write("Người dùng nhập thông tin bệnh nhân để mô hình dự đoán nguy cơ tái nhập viện.")
    st.markdown('</div>', unsafe_allow_html=True)

    X_raw, _, _, _, _ = prepare_data(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧾 Thiết kế giao diện nhập liệu")

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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Xử lý logic")
    st.write("Dữ liệu đầu vào sẽ được tiền xử lý giống như lúc huấn luyện, sau đó đưa vào mô hình LightGBM để dự đoán xác suất tái nhập viện.")

    if st.button("🔍 Dự đoán"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= best_threshold else 0

        st.markdown("### 🎯 Kết quả dự đoán")
        if pred == 1:
            st.markdown(
                f'<div class="note-orange"><b>Kết luận:</b> Bệnh nhân có nguy cơ tái nhập viện.<br><b>Xác suất dự đoán:</b> {prob:.4f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="note-green"><b>Kết luận:</b> Bệnh nhân có nguy cơ thấp.<br><b>Xác suất dự đoán:</b> {prob:.4f}</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TRANG 3
# =========================
else:
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📈 Đánh giá & Hiệu năng")

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

    st.markdown("### 📊 Các chỉ số đánh giá")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{pre:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-score", f"{f1:.4f}")
    m5.metric("ROC-AUC", f"{auc:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧩 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title("Confusion Matrix", fontsize=18, pad=10)
        ax.set_xlabel("Dự đoán", fontsize=13)
        ax.set_ylabel("Thực tế", fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"], fontsize=12)
        ax.set_yticklabels(["0", "1"], fontsize=12)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=17, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#111827"
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Báo cáo phân loại")
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Phân tích sai số và hướng cải thiện")
    st.markdown(
        f"""
        <div class="note-blue">
        Mô hình hiện tại đạt <b>Accuracy = {acc:.4f}</b>, <b>Precision = {pre:.4f}</b>,
        <b>Recall = {rec:.4f}</b>, <b>F1-score = {f1:.4f}</b>. Kết quả cho thấy mô hình
        nhận diện khá tốt các trường hợp tái nhập viện, đặc biệt Recall cao giúp hạn chế
        bỏ sót bệnh nhân cần theo dõi. Tuy nhiên Accuracy chưa cao do vẫn còn dự đoán nhầm giữa hai lớp.
        Có thể cải thiện thêm bằng cách bổ sung đặc trưng, thử XGBoost/CatBoost hoặc tinh chỉnh threshold sâu hơn.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Xem tham số tốt nhất của mô hình"):
        st.json(best_params)

    st.markdown(
        f'<div class="small-muted">Best CV F1-score trong quá trình tìm tham số: <b>{best_cv_f1:.4f}</b></div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)