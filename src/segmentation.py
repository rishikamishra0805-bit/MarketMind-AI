
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["Recency", "Frequency", "Monetary"]


def assign_segment_names(data):
    """Assign business-friendly names using average RFM behaviour."""

    data = data.copy()

    profiles = data.groupby("Cluster")[FEATURES].mean()

    median_recency = profiles["Recency"].median()
    median_frequency = profiles["Frequency"].median()
    median_monetary = profiles["Monetary"].median()

    segment_names = {}

    for cluster, row in profiles.iterrows():

        recent = row["Recency"] <= median_recency
        frequent = row["Frequency"] >= median_frequency
        high_spending = row["Monetary"] >= median_monetary

        if recent and frequent and high_spending:
            name = "Champions"

        elif not recent and not frequent and not high_spending:
            name = "At-Risk Customers"

        elif high_spending and not frequent:
            name = "Big Spenders"

        elif frequent:
            name = "Loyal Customers"

        elif recent:
            name = "Recent Customers"

        else:
            name = "Potential Customers"

        segment_names[cluster] = name

    data["Segment"] = data["Cluster"].map(segment_names)

    return data


def run_segmentation(rfm, n_clusters=4):
    """Train K-Means and return segmented customers, model, scaler and score."""

    if rfm.empty:
        raise ValueError("RFM dataset is empty. Cannot perform segmentation.")

    missing = [col for col in FEATURES if col not in rfm.columns]

    if missing:
        raise ValueError(
            "Missing RFM features: " + ", ".join(missing)
        )

    if not isinstance(n_clusters, int):
        raise ValueError("Number of clusters must be an integer.")

    if n_clusters < 2 or n_clusters >= len(rfm):
        raise ValueError(
            "Number of clusters must be at least 2 and smaller than the number of customers."
        )

    data = rfm.copy()

    # Convert RFM features to numeric values.
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.dropna(subset=FEATURES)

    if len(data) <= n_clusters:
        raise ValueError("Not enough valid customers for clustering.")

    if (data[FEATURES] < 0).any().any():
        raise ValueError("RFM values cannot be negative.")

    if not np.isfinite(data[FEATURES].to_numpy()).all():
        raise ValueError("RFM data contains invalid numeric values.")

    # Log transformation reduces the effect of highly skewed values.
    x = np.log1p(data[FEATURES])

    # Standardize features so they have comparable scales.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train the K-Means clustering model.
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(x_scaled)

    data["Cluster"] = labels

    # Silhouette Score measures how well-separated the clusters are.
    unique_labels = len(np.unique(labels))

    if 1 < unique_labels < len(data):
        score = silhouette_score(x_scaled, labels)
    else:
        score = 0.0

    # Add readable business names to the clusters.
    data = assign_segment_names(data)

    return data, model, scaler, score
