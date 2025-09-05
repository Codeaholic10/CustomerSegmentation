import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    return df

@st.cache_data
def calculate_rfm(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_data = df.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm_data.rename(columns={'InvoiceDate': 'Recency',
                             'InvoiceNo': 'Frequency',
                             'TotalPrice': 'MonetaryValue'}, inplace=True)
    return rfm_data

def run_kmeans_and_pca(rfm_data, k):
    rfm_log = np.log1p(rfm_data)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(rfm_scaled)
    clustered_rfm = rfm_data.copy()
    clustered_rfm['Cluster'] = cluster_labels
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    return clustered_rfm, pca_df

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.title("Interactive Customer Segmentation Dashboard 🛍️")
st.write("This dashboard uses RFM analysis and K-Means clustering to segment customers from an e-commerce dataset.")

df = load_and_clean_data('Online Retail.xlsx')
rfm_df = calculate_rfm(df)

st.sidebar.header("Segmentation Controls")
k_clusters = st.sidebar.slider(
    label="Select the Number of Clusters (k)",
    min_value=2,
    max_value=10,
    value=4,
    help="Choose the number of customer segments you want to create."
)

clustered_rfm, pca_df = run_kmeans_and_pca(rfm_df, k_clusters)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Customer Segments Visualization (k={k_clusters})")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Cluster',
        data=pca_df,
        palette='viridis',
        s=70,
        alpha=0.8,
        ax=ax
    )
    ax.set_title("Customer Segments in 2D PCA Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    st.pyplot(fig)

with col2:
    st.subheader("Cluster Profiles")
    st.write("Mean RFM values for each customer segment.")
    cluster_profiles = clustered_rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).round(2)
    st.dataframe(cluster_profiles, use_container_width=True)

st.subheader("Inspect Individual Clusters")
selected_cluster = st.selectbox(
    "Choose a cluster to see its customers",
    options=sorted(clustered_rfm['Cluster'].unique())
)

st.write(f"Displaying customers in Cluster {selected_cluster}")

cluster_customers = clustered_rfm[clustered_rfm['Cluster'] == selected_cluster].sort_values(
    by='MonetaryValue', ascending=False
)

st.dataframe(cluster_customers, use_container_width=True)

if st.checkbox("Show raw transactional data"):
    st.subheader("Raw Data")
    st.dataframe(df)
