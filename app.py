"""Streamlit app: mushroom edibility classifier.

Trains a random forest on the UCI Mushroom dataset (downloaded on first
run), then lets you describe a mushroom in the sidebar and predicts
whether it is edible or poisonous.

Run with:  streamlit run app.py

Educational project only - never use it to decide whether a real
mushroom is safe to eat.
"""

import os
import urllib.request

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

COLUMNS = "class,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat".split(",")

# Short labels for the features shown in the sidebar (most useful ones)
FEATURE_LABELS = {
    "odor": "Odor",
    "gill-size": "Gill size",
    "gill-color": "Gill color",
    "spore-print-color": "Spore print color",
    "bruises": "Bruises",
    "cap-color": "Cap color",
    "stalk-surface-above-ring": "Stalk surface above ring",
    "population": "Population",
}


@st.cache_data
def load_data():
    if not os.path.exists("mushrooms.csv"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        raw = urllib.request.urlopen(url).read().decode().strip().split("\n")
        with open("mushrooms.csv", "w") as f:
            f.write(",".join(COLUMNS) + "\n")
            f.write("\n".join(raw) + "\n")
    return pd.read_csv("mushrooms.csv")


@st.cache_resource
def train_model(df):
    y = (df["class"] == "p").astype(int)
    X = pd.get_dummies(df.drop(columns=["class"]))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean()
    return model, X.columns, acc, cv


df = load_data()
model, feature_names, test_acc, cv_acc = train_model(df)

st.title("Mushroom edibility classifier")
st.warning("Educational project. Never use this to decide whether a real "
           "mushroom is safe to eat.")

st.write(f"Test accuracy: **{test_acc:.4f}**  |  5-fold cross-validation: "
         f"**{cv_acc:.4f}**  (UCI Mushroom dataset, {len(df)} rows)")

# Sidebar: describe a mushroom
st.sidebar.header("Describe the mushroom")
modes = df.drop(columns=["class"]).mode().iloc[0]  # most frequent value per feature
user = dict(modes)
for col, label in FEATURE_LABELS.items():
    options = sorted(df[col].unique())
    user[col] = st.sidebar.selectbox(label, options)

row = pd.DataFrame([user])
row_enc = pd.get_dummies(row).reindex(columns=feature_names, fill_value=0)
proba = model.predict_proba(row_enc)[0]
p_poison = proba[1]

st.subheader("Prediction")
if p_poison >= 0.5:
    st.error(f"Predicted: POISONOUS (probability {p_poison:.2%})")
else:
    st.success(f"Predicted: edible (probability poison {p_poison:.2%})")

st.subheader("Most important features")
imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(10)
st.bar_chart(imp)

with st.expander("Dataset sample"):
    st.dataframe(df.head(100))
