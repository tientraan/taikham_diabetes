import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Dự đoán tái nhập viện", layout="wide")

st.title("🏥 Phân loại nguy cơ tái nhập viện (Random Forest)")

# ===== PATH =====
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    return pd.read_csv("data/diabetic_data.csv")

# ===== TRAIN =====
@st.cache_resource
def train_model(df):
    data = df.copy()

    # xử lý dữ liệu
    data = data.replace("?", pd.NA)
    data["target"] = data["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # bỏ cột không cần
    data = data.drop(columns=["encounter_id", "patient_nbr", "readmitted"])

    X = data.drop("target", axis=1)
    y = data["target"]

    # chia loại cột
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # pipeline
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # ===== RANDOM FOREST (tối ưu sẵn) =====
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # lưu model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return model, preprocessor, X_test, y_test

# ===== LOAD MODEL =====
@st.cache_resource
def load_or_train(df):
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        pre = joblib.load(PREPROCESSOR_PATH)

        data = df.copy()
        data = data.replace("?", pd.NA)
        data["target"] = data["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
        data = data.drop(columns=["encounter_id", "patient_nbr", "readmitted"])

        X = data.drop("target", axis=1)
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test = pre.transform(X_test)

        return model, pre, X, X_test, y_test

    model, pre, X_test, y_test = train_model(df)
    data = df.copy()
    data = data.replace("?", pd.NA)
    data["target"] = data["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    data = data.drop(columns=["encounter_id", "patient_nbr", "readmitted"])
    X = data.drop("target", axis=1)

    return model, pre, X, X_test, y_test

# ===== MAIN =====
df = load_data()
model, preprocessor, X, X_test, y_test = load_or_train(df)

y_pred = model.predict(X_test)

# ===== MENU =====
page = st.sidebar.radio("Chọn trang", [
    "Trang 1: EDA",
    "Trang 2: Dự đoán",
    "Trang 3: Đánh giá"
])

# ===== PAGE 1 =====
if page == "Trang 1: EDA":
    st.header("📊 Khám phá dữ liệu")

    st.write(df.head())

    st.subheader("Phân bố readmitted")
    fig, ax = plt.subplots()
    df["readmitted"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ===== PAGE 2 =====
elif page == "Trang 2: Dự đoán":
    st.header("🔮 Nhập dữ liệu")

    input_data = {}
    for col in X.columns:
        if X[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].dropna().unique())
        else:
            input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])

    if st.button("Dự đoán"):
        X_input = preprocessor.transform(input_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        if pred == 1:
            st.error(f"Nguy cơ CAO ({prob:.2f})")
        else:
            st.success(f"Nguy cơ THẤP ({prob:.2f})")

# ===== PAGE 3 =====
else:
    st.header("📈 Đánh giá mô hình")

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{pre:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1", f"{f1:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)