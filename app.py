import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Credit Card Default Prediction App")

# Load models
log_model = joblib.load("models/logistic_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
nb_model = joblib.load("models/naive_bayes_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Model dictionary
models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data")
    st.dataframe(data.head())

    # Separate features and target
    if "default.payment.next.month" in data.columns:
        X = data.drop("default.payment.next.month", axis=1)
        y = data["default.payment.next.month"]
    else:
        st.error("Target column 'default.payment.next.month' not found.")
        st.stop()

    # Model selection
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]

    # Scale if required
    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    # Predict
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)