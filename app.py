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


# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Ứng dụng phân loại tái nhập viện",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background: #f6f8fb;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    h1, h2, h3 {
        color: #183153;
    }

    .app-title {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
        padding: 22px 28px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    .info-card {
        background: white;
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border: 1px solid #e8edf4;
        margin-bottom: 12px;
    }

    .metric-box {
        background: white;
        padding: 12px;
        border-radius: 14px;
        border: 1px solid #e8edf4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        text-align: center;
    }

    .stMetric {
        background: white;
        padding: 10px;
        border-radius: 14px;
        border: 1px solid #e8edf4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
    }

    .stButton > button:hover {
        color: white;
        opacity: 0.95;
    }

    .sidebar-note {
        font-size: 14px;
        color: #475569;
        line-height: 1.6;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e8edf4;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# THÔNG TIN ĐỀ TÀI
# =========================================================
TITLE = "Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường dựa trên dữ liệu bệnh án bằng Light Gradient Boosting Machine nhằm hỗ trợ phát hiện sớm bệnh nhân có rủi ro cao"
STUDENT_NAME = "Trần Quang Tiến"
STUDENT_ID = "22T1020760"
TOPIC_DESC = (
    "Hỗ trợ phân loại bệnh nhân có nguy cơ tái nhập viện dựa trên dữ liệu bệnh án, "
    "giúp sàng lọc sớm các trường hợp rủi ro cao để ưu tiên theo dõi và hỗ trợ quyết định."
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_best.pkl")
DATA_PATH = "data/diabetic_data.csv"


# =========================================================
# CACHE DỮ LIỆU
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# =========================================================
# TIỀN XỬ LÝ
# =========================================================
def prepare_data(df):
    data = df.copy()
    data = data.replace("?", np.nan)

    # target nhị phân
    # 0 = NO, 1 = <30 hoặc >30
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


# =========================================================
# TÌM THRESHOLD TỐI ƯU
# =========================================================
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


# =========================================================
# TRAIN / LOAD MODEL
# =========================================================
@st.cache_resource
def train_or_load_model(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

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
        return (
            saved["pipeline"],
            X_test,
            y_test,
            X,
            saved["best_threshold"],
            saved["best_params"],
            saved["best_cv_f1"],
            saved["feature_cols"],
            cat_cols,
            num_cols
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
        "model__n_estimators": [300, 500],
        "model__learning_rate": [0.03, 0.05],
        "model__num_leaves": [31, 63],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=5,
        scoring="f1",
        cv=3,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val, y_val_prob)

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "best_threshold": best_threshold,
            "best_params": search.best_params_,
            "best_cv_f1": search.best_score_,
            "feature_cols": X.columns.tolist()
        },
        MODEL_PATH
    )

    return (
        best_pipeline,
        X_test,
        y_test,
        X,
        best_threshold,
        search.best_params_,
        search.best_score_,
        X.columns.tolist(),
        cat_cols,
        num_cols
    )


def reset_model():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    st.cache_resource.clear()
    st.success("Đã xóa model cũ. Hãy tải lại trang để huấn luyện lại.")


# =========================================================
# LOAD DỮ LIỆU / MODEL
# =========================================================
df = load_data()
pipeline, X_test, y_test, X_full, best_threshold, best_params, best_cv_f1, feature_cols, cat_cols, num_cols = train_or_load_model(df)

eda_data = df.copy().replace("?", np.nan)
eda_data["target"] = eda_data["readmitted"].apply(lambda x: 0 if x == "NO" else 1)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧭 Điều hướng")
    page = st.radio(
        "Chọn trang",
        [
            "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)",
            "Trang 2: Triển khai mô hình",
            "Trang 3: Đánh giá & Hiệu năng"
        ]
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="sidebar-note">
        <b>Đề tài:</b><br>{TITLE}<br><br>
        <b>Sinh viên:</b><br>{STUDENT_NAME}<br>
        <b>MSSV:</b><br>{STUDENT_ID}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    if st.button("🔄 Xóa model cũ và train lại"):
        reset_model()


# =========================================================
# HEADER CHUNG
# =========================================================
st.markdown(
    """
    <div class="app-title">
        <h1 style="margin:0;">🏥 Phân loại nguy cơ tái nhập viện với Streamlit</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TRANG 1: GIỚI THIỆU & EDA
# =========================================================
if page == "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)":
    st.header("📘 Trang 1: Giới thiệu & Khám phá dữ liệu")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Thông tin bắt buộc</h3>
                <p><b>Tên đề tài:</b> {TITLE}</p>
                <p><b>Họ tên sinh viên:</b> {STUDENT_NAME}</p>
                <p><b>MSSV:</b> {STUDENT_ID}</p>
                <p><b>Mô tả ngắn gọn giá trị thực tiễn:</b> {TOPIC_DESC}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        missing_total = int(eda_data.isna().sum().sum())
        st.markdown("### 📌 Tóm tắt dữ liệu")
        m1, m2 = st.columns(2)
        m1.metric("Số dòng", f"{eda_data.shape[0]:,}")
        m2.metric("Số cột", f"{eda_data.shape[1]:,}")
        m3, m4 = st.columns(2)
        m3.metric("Giá trị thiếu", f"{missing_total:,}")
        m4.metric("Số đặc trưng dùng", f"{len(feature_cols)}")

    st.subheader("📋 Hiển thị dữ liệu thô")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("📊 Các biểu đồ phân tích")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Biểu đồ 1: Phân bố biến mục tiêu `target`**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        eda_data["target"].value_counts().sort_index().plot(kind="bar", ax=ax1)
        ax1.set_title("Phân bố target (0 = Không, 1 = Có tái nhập viện)")
        ax1.set_xlabel("Target")
        ax1.set_ylabel("Số lượng")
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**Biểu đồ 2: Top 10 cột có nhiều giá trị thiếu**")
        missing_by_col = eda_data.isna().sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        missing_by_col.plot(kind="bar", ax=ax2)
        ax2.set_title("Top 10 cột có nhiều giá trị thiếu")
        ax2.set_xlabel("Cột")
        ax2.set_ylabel("Số lượng thiếu")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    st.markdown("**Biểu đồ 3: Phân bố số ngày nằm viện**")   
    with col_center:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        eda_data["time_in_hospital"].dropna().astype(float).plot(kind="hist", bins=20, ax=ax3)
        ax3.set_title("Phân bố time_in_hospital")
        ax3.set_xlabel("Số ngày")
        ax3.set_ylabel("Tần suất")
        plt.tight_layout()
        st.pyplot(fig3)


    st.subheader("📝 Giải thích / Nhận xét dữ liệu")
    class_counts = eda_data["target"].value_counts()
    class_ratio_1 = (class_counts.get(1, 0) / len(eda_data)) * 100
    class_ratio_0 = (class_counts.get(0, 0) / len(eda_data)) * 100

    st.write(
        f"""
        - Dữ liệu gồm nhiều thông tin về đặc điểm bệnh nhân, xét nghiệm, thuốc điều trị và lịch sử nhập viện.
        - Biến `readmitted` ban đầu đã được chuyển thành **biến mục tiêu nhị phân**:
          **0 = không tái nhập viện**, **1 = có tái nhập viện**.
        - Tập dữ liệu có cả biến số và biến phân loại nên cần xử lý thiếu dữ liệu và mã hóa trước khi đưa vào mô hình.
        - Tỷ lệ lớp hiện tại cho thấy dữ liệu **không cân bằng hoàn toàn**:
          lớp 0 chiếm khoảng **{class_ratio_0:.2f}%**, lớp 1 chiếm khoảng **{class_ratio_1:.2f}%**.
        - Một số đặc trưng có khả năng liên quan đến rủi ro tái nhập viện là:
          **time_in_hospital, num_medications, number_inpatient, number_emergency, number_diagnoses**.
        """
    )


# =========================================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# =========================================================
elif page == "Trang 2: Triển khai mô hình":
    st.header("🛠️ Trang 2: Triển khai mô hình")

    info1, info2 = st.columns([1.2, 1])
    with info1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Thiết kế giao diện nhập liệu</h3>
                <p>Người dùng nhập thông tin bệnh nhân bằng các widget như <b>selectbox</b> và <b>number_input</b>.</p>
                <p>Dữ liệu đầu vào sẽ được tiền xử lý giống hệt lúc huấn luyện trước khi dự đoán.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info2:
        st.metric("Ngưỡng dự đoán đang dùng", f"{best_threshold:.2f}")
        st.metric("Số đặc trưng nhập liệu", len(feature_cols))

    X_raw, _, cat_cols2, num_cols2, _ = prepare_data(df)
    X_raw = X_raw[feature_cols].copy()

    st.subheader("🧾 Nhập thông tin bệnh nhân")

    input_data = {}
    col_left, col_right = st.columns(2)
    columns = X_raw.columns.tolist()

    for i, col in enumerate(columns):
        current_col = col_left if i % 2 == 0 else col_right

        with current_col:
            if col in cat_cols2:
                options = sorted(X_raw[col].dropna().astype(str).unique().tolist())

                if len(options) == 0:
                    input_data[col] = ""
                    st.text_input(col, value="", disabled=True)
                else:
                    input_data[col] = st.selectbox(
                        label=col,
                        options=options,
                        index=0
                    )
            else:
                series = pd.to_numeric(X_raw[col], errors="coerce").dropna()

                if series.empty:
                    input_data[col] = 0.0
                    st.number_input(col, value=0.0)
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

    input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols)

    for col in cat_cols2:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    for col in num_cols2:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    st.subheader("🔍 Dữ liệu đầu vào sau khi gom thành bảng")
    st.dataframe(input_df, use_container_width=True)

    st.subheader("⚙️ Xử lý logic")
    st.write(
        """
        - Mô hình đã được load từ file đã huấn luyện.
        - Dữ liệu đầu vào được tiền xử lý theo đúng pipeline huấn luyện.
        - Kết quả đầu ra gồm:
          - Nhãn dự đoán
          - Xác suất / độ tin cậy của dự đoán
        """
    )

    if st.button("📌 Dự đoán nguy cơ tái nhập viện"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= best_threshold else 0

        st.subheader("✅ Kết quả dự đoán")

        k1, k2 = st.columns(2)
        with k1:
            if pred == 1:
                st.error("Kết luận: Bệnh nhân **có nguy cơ tái nhập viện**.")
            else:
                st.success("Kết luận: Bệnh nhân **có nguy cơ thấp**.")

        with k2:
            st.info(f"Độ tin cậy / Xác suất dự đoán: **{prob:.4f}**")

        st.progress(float(prob))


# =========================================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# =========================================================
else:
    st.header("📈 Trang 3: Đánh giá & Hiệu năng")

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    use_auto = st.checkbox("Dùng threshold tối ưu", value=True)
    if use_auto:
        threshold = best_threshold
    else:
        threshold = st.slider("Chọn threshold", 0.10, 0.90, 0.50, 0.01)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("📏 Các chỉ số đo lường")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{pre:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-score", f"{f1:.4f}")
    m5.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("🧩 Biểu đồ kỹ thuật")
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.8))
        ax_cm.imshow(cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Dự đoán")
        ax_cm.set_ylabel("Thực tế")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["0", "1"])
        ax_cm.set_yticklabels(["0", "1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)

        plt.tight_layout()
        st.pyplot(fig_cm)

    with chart2:
        st.markdown("**Phân bố xác suất dự đoán**")
        fig_prob, ax_prob = plt.subplots(figsize=(5.5, 4.8))
        ax_prob.hist(y_prob, bins=25)
        ax_prob.set_title("Histogram xác suất dự đoán")
        ax_prob.set_xlabel("Xác suất lớp 1")
        ax_prob.set_ylabel("Tần suất")
        plt.tight_layout()
        st.pyplot(fig_prob)

    st.subheader("📄 Báo cáo phân loại")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("🔎 Phân tích sai số")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_errors = fp + fn

    err1, err2, err3 = st.columns(3)
    err1.metric("Dự đoán sai tổng", int(total_errors))
    err2.metric("False Positive", int(fp))
    err3.metric("False Negative", int(fn))

    st.write(
        f"""
        - Mô hình hiện tại đạt **Accuracy = {acc:.4f}**, **Precision = {pre:.4f}**, **Recall = {rec:.4f}**, **F1-score = {f1:.4f}**.
        - **False Negative = {fn}** là số bệnh nhân thực sự có nguy cơ nhưng mô hình dự đoán thấp, đây là nhóm sai số cần chú ý nhất.
        - **False Positive = {fp}** là số trường hợp mô hình cảnh báo rủi ro nhưng thực tế không thuộc lớp nguy cơ.
        - Với bài toán hỗ trợ phát hiện sớm bệnh nhân rủi ro, cần ưu tiên hạn chế bỏ sót nên chỉ số **Recall** rất quan trọng.
        - Có thể cải thiện thêm bằng cách:
          1. Điều chỉnh threshold phù hợp mục tiêu,
          2. Chọn thêm đặc trưng liên quan mạnh hơn,
          3. Làm sạch dữ liệu thiếu tốt hơn,
          4. Cân bằng dữ liệu hoặc tinh chỉnh siêu tham số sâu hơn.
        """
    )

    with st.expander("Xem tham số tốt nhất của mô hình"):
        st.json(best_params)

    st.info(f"Best CV F1-score trong quá trình tìm tham số: **{best_cv_f1:.4f}**")