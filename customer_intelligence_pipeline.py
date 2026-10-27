import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Step 01: Generate Synthetic E-Commerce RFM Data
np.random.seed(42)
n_customers = 1000

data = {
    'CustomerID': [f'CUST_{i:04d}' for i in range(n_customers)],
    'Recency': np.random.exponential(scale=30, size=n_customers).astype(int) + 1,  
    'Frequency': np.random.poisson(lam=5, size=n_customers) + 1,                   
    'Monetary': np.random.lognormal(mean=5.5, sigma=0.8, size=n_customers).round(2)  
}

rfm_df = pd.DataFrame(data)

print("--- Raw RFM Data Sample ---")
print(rfm_df.head(3))
print(f"\nTotal Customers Loaded: {len(rfm_df)}\n" + "="*50 + "\n")

#Step 02: Feature Scaling
scaler = StandardScaler()
features = ['Recency', 'Frequency', 'Monetary']
rfm_scaled = scaler.fit_transform(rfm_df[features])

#step 03: Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

rfm_df['PCA1'] = rfm_pca[:, 0]
rfm_df['PCA2'] = rfm_pca[:, 1]

explained_variance = pca.explained_variance_ratio_.sum() * 100
print(f"PCA Variance Explained by 2 Components: {explained_variance:.2f}%")


#step 04: Unsupervised Clustering using K-means
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

sil_score = silhouette_score(rfm_scaled, rfm_df['Cluster'])
print(f"K-Means Clustering Silhouette Score: {sil_score:.3f}")

#Step 05: Anomaly & Outlier Detection
iso = IsolationForest(contamination=0.03, random_state=42)
rfm_df['Anomaly_Flag'] = iso.fit_predict(rfm_scaled)

rfm_df['Is_Outlier'] = rfm_df['Anomaly_Flag'] == -1
num_outliers = rfm_df['Is_Outlier'].sum()
print(f"Isolation Forest flagged {num_outliers} anomalous accounts.\n" + "="*50 + "\n")

#Step 06: Business Persona Summaries
print("--- Cluster Persona Profiles ---")
cluster_summary = rfm_df.groupby('Cluster')[features].mean().reset_index()
cluster_summary['Customer_Count'] = rfm_df.groupby('Cluster')['CustomerID'].count().values

#Map intuitive business personas
persona_mapping = {
    0: "Standard Shoppers",
    1: "VIP Whales (High Spend/Freq)",
    2: "At-Risk / Churned Customers",
    3: "New / Occasional Buyers"
}
cluster_summary['Persona'] = cluster_summary['Cluster'].map(persona_mapping)

print(cluster_summary[['Cluster', 'Persona', 'Customer_Count', 'Recency', 'Frequency', 'Monetary']].to_string(index=False))
print("\n" + "="*50)
print("--- Sample Flagged Anomalies / Outliers ---")
print(rfm_df[rfm_df['Is_Outlier']][['CustomerID', 'Recency', 'Frequency', 'Monetary']].head(3).to_string(index=False))
