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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve
)

st.set_page_config(page_title="Phân loại tái nhập viện", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #2c3e50;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
div[data-testid="stSidebar"] {
    background-color: #eef3f8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CẤU HÌNH CHUNG
# =========================
TITLE = "Phân loại nguy cơ tái nhập viện của bệnh nhân tiểu đường dựa trên dữ liệu bệnh án bằng Random Forest nhằm hỗ trợ phát hiện sớm bệnh nhân có rủi ro cao"
STUDENT_NAME = "Họ tên sinh viên: Trần Quang Tiến"
STUDENT_ID = "MSSV: 22T1020760"
DATA_SOURCE = "UCI Machine Learning Repository - Diabetes 130-US hospitals for years 1999-2008"

TOPIC_DESC = (
    "Bài toán giúp xác định sớm bệnh nhân có nguy cơ tái nhập viện cao dựa trên dữ liệu bệnh án, "
    "từ đó hỗ trợ bác sĩ theo dõi, phân nhóm rủi ro và đưa ra phương án điều trị phù hợp hơn. "
    "Kết quả góp phần nâng cao hiệu quả điều trị, giảm chi phí y tế và hạn chế tái nhập viện không mong muốn."
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_best.pkl")


# =========================
# HÀM TẢI DỮ LIỆU
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/diabetic_data.csv")


# =========================
# TIỀN XỬ LÝ DỮ LIỆU
# =========================
def prepare_data(df):
    data = df.copy()
    data = data.replace("?", np.nan)

    # Nhãn mục tiêu: 0 = không tái nhập viện, 1 = có tái nhập viện
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
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        required_keys = [
            "pipeline",
            "best_threshold",
            "best_params",
            "best_cv_f1",
            "feature_cols"
        ]
        if all(k in saved for k in required_keys):
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
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=12,
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
            "best_val_f1": best_val_f1,
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
    st.success("Đã xóa model cũ. Tải lại trang để huấn luyện lại mô hình.")


# =========================
# FEATURE IMPORTANCE
# =========================
def get_feature_importance(pipeline, feature_cols):
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        transformed_feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": transformed_feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        return fi_df
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


# =========================
# LOAD
# =========================
df = load_data()
pipeline, X_test, y_test, X_full, best_threshold, best_params, best_cv_f1, feature_cols, cat_cols, num_cols = train_or_load_model(df)

st.markdown("""
<h1 style='color: #2c3e50;'>
🚀 Phân loại bệnh nhân tái nhập viện
</h1>
<hr>
""", unsafe_allow_html=True)

st.subheader(TITLE)

st.sidebar.title("Điều hướng")
page = st.sidebar.radio(
    "Chọn trang",
    [
        "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)",
        "Trang 2: Triển khai mô hình",
        "Trang 3: Đánh giá & Hiệu năng"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin mô hình")
st.sidebar.write("**Thuật toán:** Random Forest")
st.sidebar.write(f"**Ngưỡng tối ưu:** {best_threshold:.2f}")
st.sidebar.write(f"**Số đặc trưng đầu vào:** {len(feature_cols)}")

if st.sidebar.button("Xóa model cũ và train lại"):
    reset_model()


# =========================
# TRANG 1
# =========================
if page == "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)":
    st.header("📘 Giới thiệu đề tài")

    st.write(f"**Tên đề tài:** {TITLE}")
    st.write(f"**{STUDENT_NAME}**")
    st.write(f"**{STUDENT_ID}**")
    st.write(f"**Nguồn dữ liệu:** {DATA_SOURCE}")
    st.write(f"**Giá trị thực tiễn:** {TOPIC_DESC}")

    st.header("📊 Khám phá dữ liệu (EDA)")

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head(10), use_container_width=True)

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

    st.subheader("Biểu đồ 3: Phân bố số ngày nằm viện")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    df["time_in_hospital"].dropna().hist(ax=ax3, bins=15)
    ax3.set_title("Phân bố time_in_hospital")
    ax3.set_xlabel("Số ngày nằm viện")
    ax3.set_ylabel("Tần suất")
    st.pyplot(fig3)

    st.subheader("Nhận xét dữ liệu")
    st.write(f"""
- Bộ dữ liệu có **{df.shape[0]} dòng** và **{df.shape[1]} cột**.
- Sau khi chuyển đổi nhãn, bài toán trở thành **phân loại nhị phân**.
- Dữ liệu có hiện tượng **mất cân bằng lớp**, do số bệnh nhân không tái nhập viện thường lớn hơn nhóm tái nhập viện.
- Dữ liệu gồm cả **biến số** và **biến phân loại**, vì vậy cần tiền xử lý khác nhau trước khi huấn luyện.
- Một số đặc trưng được xem là quan trọng trong bài toán này gồm: **time_in_hospital, num_medications, number_inpatient, number_emergency, number_diagnoses, A1Cresult, insulin**.
- Dữ liệu có chứa giá trị thiếu dưới dạng **"?"**, do đó cần được xử lý trước khi đưa vào mô hình.
""")


# =========================
# TRANG 2
# =========================
elif page == "Trang 2: Triển khai mô hình":
    st.header("🛠️ Triển khai mô hình")

    st.subheader("Mô tả mô hình")
    st.write("""
Quy trình xử lý:
1. Người dùng nhập thông tin bệnh nhân.
2. Dữ liệu đầu vào được chuẩn hóa kiểu dữ liệu giống lúc huấn luyện.
3. Mô hình Random Forest đã huấn luyện trước sẽ được load từ file `.pkl`.
4. Input được đưa qua pipeline gồm:
   - xử lý giá trị thiếu
   - mã hóa One-Hot cho biến phân loại
   - dự đoán bằng Random Forest
5. Hệ thống trả về:
   - nhãn dự đoán
   - xác suất bệnh nhân có nguy cơ tái nhập viện
""")

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
                    input_data[col] = st.selectbox(col, options, index=default_index)
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

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols)

    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    for col in num_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    st.subheader("Dữ liệu đầu vào sau xử lý")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Dự đoán"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = 1 if prob >= best_threshold else 0

        if prob >= 0.7:
            risk_level = "Nguy cơ cao"
        elif prob >= 0.4:
            risk_level = "Nguy cơ trung bình"
        else:
            risk_level = "Nguy cơ thấp"

        st.subheader("Kết quả dự đoán")

        if pred == 1:
            st.error("Kết luận: Bệnh nhân **có nguy cơ tái nhập viện**.")
        else:
            st.success("Kết luận: Bệnh nhân **có nguy cơ thấp**.")

        st.info(f"Xác suất tái nhập viện: **{prob:.4f}**")
        st.write(f"**Mức độ nguy cơ:** {risk_level}")


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
        threshold = st.slider("Threshold", 0.10, 0.90, 0.50, 0.01)

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

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
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
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("Báo cáo phân loại")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    st.subheader("Feature Importance")
    fi_df = get_feature_importance(pipeline, feature_cols)

    if not fi_df.empty:
        top_fi = fi_df.head(10).sort_values(by="importance", ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        ax_fi.barh(top_fi["feature"], top_fi["importance"])
        ax_fi.set_title("Top 10 đặc trưng quan trọng")
        ax_fi.set_xlabel("Importance")
        st.pyplot(fig_fi)
    else:
        st.warning("Không thể hiển thị feature importance.")

    tn, fp, fn, tp = cm.ravel()

    st.subheader("Phân tích sai số và hướng cải thiện")
    st.write(f"""
- **False Positive (FP): {fp}** → mô hình dự đoán bệnh nhân có nguy cơ tái nhập viện nhưng thực tế không tái nhập viện.
- **False Negative (FN): {fn}** → mô hình bỏ sót bệnh nhân có nguy cơ tái nhập viện. Đây là loại sai số cần hạn chế nhất trong bài toán y tế.
- Mô hình hiện tại đạt **Accuracy = {acc:.4f}**, **Precision = {pre:.4f}**, **Recall = {rec:.4f}**, **F1-score = {f1:.4f}**, **ROC-AUC = {auc:.4f}**.
- Recall càng cao thì mô hình càng phát hiện tốt các bệnh nhân có nguy cơ cao.
- Precision phản ánh mức độ chính xác của các cảnh báo mà mô hình đưa ra.
- Có thể cải thiện thêm bằng cách:
  1. chọn thêm đặc trưng phù hợp,
  2. xử lý mất cân bằng dữ liệu tốt hơn,
  3. tối ưu siêu tham số sâu hơn,
  4. điều chỉnh threshold phù hợp mục tiêu bài toán.
""")

    with st.expander("Xem tham số tốt nhất của mô hình"):
        st.json(best_params)

    st.write(f"**Best CV F1-score trong quá trình tìm tham số:** {best_cv_f1:.4f}")