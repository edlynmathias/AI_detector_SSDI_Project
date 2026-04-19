import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, roc_curve, auc
from scipy.stats import norm

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    layout="wide"
)

st.title("AI Generated vs Real Image Detector")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_cnn():
    model = load_model("my_model.h5")
    return model

model = load_cnn()

IMG_SIZE = (32,32)

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# TABS
# ----------------------------
tab1, tab2 = st.tabs(["Image Prediction","Statistical Analysis"])

# ============================================================
# TAB 1 : IMAGE PREDICTION
# ============================================================

with tab1:

    st.header("Upload Image")

    uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded_file:

        img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image")

        img_input = preprocess(img)

        prediction = model.predict(img_input)[0][0]

        if prediction > 0.5:
            label = "REAL"
        else:
            label = "AI GENERATED"

        with col2:

            st.subheader("Prediction")

            st.metric("Result", label)

            st.metric("Confidence", f"{prediction:.4f}")

# ============================================================
# TAB 2 : MODEL STATISTICS
# ============================================================

with tab2:

    st.header("Model Evaluation")

    # LOAD CSV DATA
    real_df = pd.read_csv("predictions_real.csv")
    ai_df = pd.read_csv("predictions_ai.csv")

    real_df["true_label"] = 0
    ai_df["true_label"] = 1

    df = pd.concat([real_df, ai_df], ignore_index=True)

    df["prediction"] = df["prediction"].map({
        "REAL":0,
        "AI":1
    })

    y_true = df["true_label"]
    y_pred = df["prediction"]

    # ----------------------------
    # METRICS
    # ----------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy", round(accuracy,3))
    col2.metric("Precision", round(precision,3))
    col3.metric("Recall", round(recall,3))
    col4.metric("F1 Score", round(f1,3))

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true,y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=["REAL","AI"],
                yticklabels=["REAL","AI"],
                ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    st.pyplot(fig)

    # ----------------------------
    # MLE ESTIMATION
    # ----------------------------

    st.subheader("Maximum Likelihood Estimation")

    n = len(y_true)

    correct_predictions = (y_true == y_pred).sum()

    p_hat = correct_predictions / n

    st.write("Total Samples:", n)
    st.write("Correct Predictions:", correct_predictions)

    st.success(f"MLE Estimate of Accuracy = {p_hat:.4f}")

    # ----------------------------
    # HYPOTHESIS TESTING
    # ----------------------------

    st.subheader("Hypothesis Testing")

    errors = (y_true != y_pred).sum()

    e_hat = errors / n

    e0 = 0.5

    z = (e_hat - e0) / np.sqrt((e0*(1-e0))/n)

    alpha = 0.05

    z_critical = norm.ppf(1-alpha/2)

    p_value = 2*(1-norm.cdf(abs(z)))

    st.write("Observed Error Rate:", e_hat)
    st.write("Z Statistic:", z)
    st.write("Z Critical:", z_critical)
    st.write("P Value:", p_value)

    if abs(z) > z_critical:

        if z < 0:

            st.success("Reject H0 : Model error is significantly lower than random guessing")

        else:

            st.error("Reject H0 : Model error is worse than random guessing")

    else:

        st.warning("Fail to reject H0 : No significant difference")

    # ----------------------------
    # CLASSIFICATION REPORT
    # ----------------------------

    st.subheader("Classification Report")

    report = classification_report(y_true,y_pred, target_names=["REAL","AI"])

    st.text(report)

    # ----------------------------
    # ROC CURVE
    # ----------------------------

    st.subheader("ROC Curve")

    y_prob = df["probability"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    roc_auc = auc(fpr,tpr)

    fig2, ax2 = plt.subplots()

    ax2.plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}")

    ax2.plot([0,1],[0,1],'--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")

    ax2.legend()

    st.pyplot(fig2)