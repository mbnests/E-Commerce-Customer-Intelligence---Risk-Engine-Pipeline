# E-Commerce Customer Intelligence & Risk Engine

An end-to-end unsupervised machine learning pipeline designed to segment unlabelled e-commerce customer bases, visualize high-dimensional behavior, and detect financial anomalies or fraudulent accounts.

---

# Project Overview
In real-world e-commerce and retail environments, customer transaction data rarely comes with pre-labeled categories. This project leverages **unsupervised machine learning** to extract actionable business intelligence without needing historical target labels:
1. **Behavioral Feature Engineering:** Converts raw transaction logs into the classic **RFM (Recency, Frequency, Monetary)** framework.
2. **Dimensionality Reduction (PCA):** Compresses multi-dimensional space to optimize variance and prepare features for distance-based algorithms.
3. **Customer Segmentation (K-Means):** Discovers natural buyer groupings and maps them to clear business personas (e.g., VIPs, At-Risk, New Buyers).
4. **Anomaly & Outlier Detection (Isolation Forest):** Automatically flags high-risk accounts, automated scraper bots, or high-value wholesale anomalies.

---

# 🏗️ Pipeline Architecture

```text
Raw Transaction Logs 
       │
       ▼
RFM Feature Engineering (Recency, Frequency, Monetary)
       │
       ▼
Feature Scaling (StandardScaler)
       │
       ├──────────────────────────────┐
       ▼                              ▼
Dimensionality Reduction (PCA)   K-Means Clustering (Segmentation)
       │                              │
       ▼                              ▼
 2D Variance Mapping            Business Personas
       │
       └──────────────────────────────┬
                                      ▼
                       Isolation Forest (Anomaly Detection)
                                      │
                                      ▼
                        Actionable Risk & Marketing Insights
