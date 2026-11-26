"""Mushroom edibility classification.

Dataset: UCI Mushroom dataset (8124 mushrooms described by 22 categorical
features; target: edible or poisonous).
All features are categorical, so we one-hot encode them, then compare a
decision tree and a random forest on a held-out test split.

Note: this dataset is known to be almost linearly separable (odor alone
separates most cases), so very high accuracy is expected and is a property
of the dataset, not of fancy modeling.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the data (download from UCI if it is not here yet)
import os, urllib.request

COLUMNS = "class,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat".split(",")

if not os.path.exists("mushrooms.csv"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    print("mushrooms.csv not found, downloading from UCI ...")
    raw = urllib.request.urlopen(url).read().decode().strip().split("\n")
    with open("mushrooms.csv", "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        f.write("\n".join(raw) + "\n")

df = pd.read_csv("mushrooms.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print("Class balance:", dict(df["class"].value_counts()), "\n")

# 2. One-hot encode the categorical features
y = (df["class"] == "p").astype(int)  # 1 = poisonous, 0 = edible
X = pd.get_dummies(df.drop(columns=["class"]))
print(f"After one-hot encoding: {X.shape[1]} binary features")

# 3. Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train two models and compare
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:15s} test accuracy: {acc:.4f}")

# 5. Confusion matrix + top features for the random forest
rf = models["Random Forest"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, rf.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["edible", "poisonous"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Random Forest - confusion matrix")

importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
importances.sort_values().plot.barh(ax=axes[1])
axes[1].set_title("Top 10 feature importances")

plt.tight_layout()
plt.savefig("results.svg", dpi=120)
print("Saved results.svg")
