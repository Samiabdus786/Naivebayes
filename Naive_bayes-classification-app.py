import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Logger ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ---------------- Page Config ----------------
st.set_page_config("Naive Bayes - IRIS", layout="wide")
st.title(" End-to-End Naive Bayes Classification (IRIS Dataset)")

# ---------------- Sidebar ----------------
st.sidebar.header("Model Settings")
nb_type = st.sidebar.selectbox(
    "Select Naive Bayes Algorithm",
    ["GaussianNB", "MultinomialNB", "BernoulliNB"]
)

alpha = st.sidebar.slider(
    "Alpha (only for Multinomial & Bernoulli)",
    0.01, 5.0, 1.0
)

log(f"NB Selected: {nb_type}")

# ================= STEP 1: Load Dataset =================
st.header("Step 1: Load IRIS Dataset")

iris = load_iris(as_frame=True)
df = iris.frame
df["target"] = iris.target
df["target_name"] = df["target"].map(
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)

st.success("IRIS dataset loaded successfully")
st.dataframe(df.head())

# ================= STEP 2: EDA =================
st.header("Step 2: Exploratory Data Analysis")

st.write("Shape:", df.shape)
st.write("Missing Values:", df.isnull().sum())

fig, ax = plt.subplots()
sns.countplot(x="target_name", data=df, ax=ax)
ax.set_title("Class Distribution")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ================= STEP 3: Feature / Target =================
st.header("Step 3: Feature Selection")

X = df.iloc[:, :4]
y = df["target_name"]

st.success("Features and target selected")

# ================= STEP 4: Train-Test Split =================
st.header("Step 4: Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

st.info("75% Training | 25% Testing")

# ================= STEP 5: Train Naive Bayes =================
st.header("Step 5: Train Model")

# Handle model-specific preprocessing
if nb_type == "GaussianNB":
    model = GaussianNB()
    X_train_nb = X_train
    X_test_nb = X_test

elif nb_type == "MultinomialNB":
    # Multinomial requires non-negative integer values
    X_train_nb = (X_train - X_train.min()).astype(int)
    X_test_nb = (X_test - X_train.min()).astype(int)
    model = MultinomialNB(alpha=alpha)

else:  # BernoulliNB
    binarizer = Binarizer(threshold=X_train.mean().mean())
    X_train_nb = binarizer.fit_transform(X_train)
    X_test_nb = binarizer.transform(X_test)
    model = BernoulliNB(alpha=alpha)

model.fit(X_train_nb, y_train)
y_pred = model.predict(X_test_nb)

# ================= STEP 6: Evaluation =================
st.header("Step 6: Model Evaluation")

acc = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {acc:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

log("Training and evaluation completed")
