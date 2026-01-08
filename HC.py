import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="🌸 Iris Hierarchical Clustering App",
    layout="wide"
)

st.title("🌸 Iris Hierarchical Clustering App")

# -------------------------------------------------
# BASE DIRECTORY (CLOUD + LOCAL SAFE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# LOAD DATASET (CSV → FALLBACK TO SKLEARN)
# -------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "DATA_SETS", "iris_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded from CSV")
else:
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename columns to match training schema
    df = df.rename(columns={
        "sepal length (cm)": "Sepal.Length",
        "sepal width (cm)": "Sepal.Width",
        "petal length (cm)": "Petal.Length",
        "petal width (cm)": "Petal.Width"
    })

    df["Species"] = iris.target_names[iris.target]

# -------------------------------------------------
# REQUIRED COLUMNS CHECK
# -------------------------------------------------
REQUIRED_COLUMNS = [
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width",
    "Species"
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"❌ Dataset missing columns: {missing}")
    st.stop()

# -------------------------------------------------
# ENCODING + SCALING
# -------------------------------------------------
le = LabelEncoder()
df["Species_encoded"] = le.fit_transform(df["Species"])

FEATURE_COLUMNS = [
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width",
    "Species_encoded"
]

X = df[FEATURE_COLUMNS]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# HIERARCHICAL CLUSTERING (UNSUPERVISED)
# -------------------------------------------------
hc_model = AgglomerativeClustering(
    n_clusters=3,
    metric="euclidean",
    linkage="ward"
)

df["Cluster"] = hc_model.fit_predict(X_scaled)

# -------------------------------------------------
# DATASET VIEW
# -------------------------------------------------
st.subheader("📄 Clustered Dataset")
st.dataframe(
    df[
        [
            "Species",
            "Species_encoded",
            "Cluster",
            "Sepal.Length",
            "Sepal.Width",
            "Petal.Length",
            "Petal.Width"
        ]
    ],
    use_container_width=True
)

# -------------------------------------------------
# SPECIES → ENCODED → CLUSTER MAPPING
# -------------------------------------------------
st.subheader("🧬 Species → Encoded → Cluster Mapping")

mapping_df = (
    df[["Species", "Species_encoded", "Cluster"]]
    .drop_duplicates()
    .sort_values(["Species_encoded", "Cluster"])
    .reset_index(drop=True)
)

st.table(mapping_df)

# -------------------------------------------------
# CLUSTER DISTRIBUTION
# -------------------------------------------------
st.subheader("📊 Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts().sort_index())

# -------------------------------------------------
# CLUSTER VISUALIZATION
# -------------------------------------------------
st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Sepal.Length",
    y="Petal.Length",
    hue="Cluster",
    palette="viridis",
    ax=ax
)

ax.set_title("Hierarchical Clustering Result")
st.pyplot(fig)

# -------------------------------------------------
# PREDICT / ASSIGN CLUSTER FOR NEW DATA
# -------------------------------------------------
st.subheader("🔮 Assign Cluster to New Flower")

col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sw = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
    pl = st.number_input("Petal Length", 1.0, 7.0, 1.4)

with col2:
    pw = st.number_input("Petal Width", 0.1, 2.5, 0.2)
    species = st.selectbox("Species", df["Species"].unique())

if st.button("Predict Cluster"):
    species_encoded = le.transform([species])[0]

    input_df = pd.DataFrame(
        [[sl, sw, pl, pw, species_encoded]],
        columns=FEATURE_COLUMNS
    )

    input_scaled = scaler.transform(input_df)

    combined = np.vstack([X_scaled, input_scaled])
    predicted_cluster = hc_model.fit_predict(combined)[-1]

    st.success("✅ Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Species", species)
    c2.metric("Species Encoded", species_encoded)
    c3.metric("Predicted Cluster", predicted_cluster)

    result_df = input_df.copy()
    result_df.insert(0, "Species", species)
    result_df["Cluster"] = predicted_cluster

    st.dataframe(result_df, use_container_width=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Model:** Hierarchical Clustering  
**Type:** Unsupervised Learning  
**Algorithm:** Agglomerative Clustering  
**Deployment Safe:** ✅  
**CSV Optional:** ✅  
""")
