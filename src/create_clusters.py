import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def create_shading_clusters(input_dir, output_dir, verbose_output=False, create_plots=False, plots_dir=None):
    """Create shading clusters from feature vector files"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose_output:
        print(f"Reading shading features from: {input_path}")
        print(f"Writing clustering results to: {output_path}")
    
    # Load all feature vector files
    feature_files = list(input_path.glob("feature_vectors_*.csv"))
    
    if not feature_files:
        if verbose_output:
            print("No shading feature files found")
        return
    
    all_data = []
    for feature_file in feature_files:
        plant_id = feature_file.stem.replace("feature_vectors_", "")
        
        # Check if output already exists
        output_file = output_path / f"{plant_id}_shading_clusters.csv"
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: shading clusters already exist, skipping")
            continue
        
        try:
            df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            if df.empty:
                continue
            
            plant_df = df.copy()
            plant_df["plant"] = plant_id
            all_data.append(plant_df)
            
            if verbose_output:
                print(f"  Loaded {plant_id}: {len(df)} samples")
                
        except Exception as e:
            if verbose_output:
                print(f"  Error loading {plant_id}: {e}")
            continue
    
    if not all_data:
        if verbose_output:
            print("No valid feature data found")
        return
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    non_feature_cols = ["date", "plant"]
    feature_cols = [col for col in combined_data.columns if col not in non_feature_cols]
    
    cluster_input = combined_data[feature_cols].dropna()
    valid_indices = cluster_input.index
    clustering_data = combined_data.loc[valid_indices].copy()
    
    # Check if we have enough data for clustering
    if len(clustering_data) < 100:
        if verbose_output:
            print(f"Not enough data for clustering: {len(clustering_data)} samples (minimum required: 100)")
        return
    
    if verbose_output:
        print(f"Clustering {len(clustering_data)} data points from {len(feature_files)} plants")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
    
    cluster_labels = clusterer.fit_predict(cluster_input)
    clustering_data["cluster"] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    if verbose_output:
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    
    if create_plots and plots_dir:
        _create_clustering_plot(
            cluster_input.values, cluster_labels, clustering_data,
            "HDBSCAN Shading Clusters",
            plots_dir,
            "shading_clusters_pca.png"
        )
    
    # Save results for each plant
    plant_ids = clustering_data["plant"].unique()
    for plant_id in plant_ids:
        plant_mask = clustering_data["plant"] == plant_id
        plant_results = clustering_data[plant_mask].copy()
        
        if not plant_results.empty:
            plant_results = plant_results.drop(columns=["plant"])
            output_file = output_path / f"{plant_id}_clustered.csv"
            plant_results.to_csv(output_file, index=False)
            
            if verbose_output:
                print(f"    Saved {plant_id}: {len(plant_results)} clustered samples")
    
    if verbose_output:
        print("Shading clustering complete.")


def create_pollution_clusters(input_dir, output_dir, n_components=3, covariance_type="full", random_state=42, verbose_output=False, create_plots=False, plots_dir=None):
    """Create pollution clusters from feature vector files"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose_output:
        print(f"Reading pollution features from: {input_path}")
        print(f"Writing clustering results to: {output_path}")
    
    # Load all feature vector files
    feature_files = list(input_path.glob("feature_vectors_*.csv"))
    
    if not feature_files:
        if verbose_output:
            print("No pollution feature files found")
        return
    
    all_data = []
    for feature_file in feature_files:
        plant_id = feature_file.stem.replace("feature_vectors_", "")
        
        # Check if output already exists
        output_file = output_path / f"{plant_id}_clustered.csv"
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: pollution clusters already exist, skipping")
            continue
        
        try:
            df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            if df.empty:
                continue
            
            plant_df = df.copy()
            plant_df["plant"] = plant_id
            all_data.append(plant_df)
            
            if verbose_output:
                print(f"  Loaded {plant_id}: {len(df)} samples")
                
        except Exception as e:
            if verbose_output:
                print(f"  Error loading {plant_id}: {e}")
            continue
    
    if not all_data:
        if verbose_output:
            print("No valid feature data found")
        return
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    non_feature_cols = ["date", "plant"]
    feature_cols = [col for col in combined_data.columns if col not in non_feature_cols]
    
    cluster_input = combined_data[feature_cols].dropna()
    valid_indices = cluster_input.index
    clustering_data = combined_data.loc[valid_indices].copy()
    
    if verbose_output:
        print(f"Clustering {len(clustering_data)} data points from {len(feature_files)} plants")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
        n_init=10
    )
    
    cluster_labels = gmm.fit_predict(cluster_input)
    cluster_probabilities = gmm.predict_proba(cluster_input)
    
    clustering_data["cluster"] = cluster_labels
    
    for i in range(n_components):
        clustering_data[f"cluster_proba_{i}"] = cluster_probabilities[:, i]
    
    if verbose_output:
        print(f"GMM: {n_components} components")
    
    if create_plots and plots_dir:
        _create_clustering_plot(
            cluster_input.values, cluster_labels, clustering_data,
            "GMM Pollution Clusters",
            plots_dir,
            "pollution_clusters_pca.png"
        )
    
    # Save results for each plant
    plant_ids = clustering_data["plant"].unique()
    for plant_id in plant_ids:
        plant_mask = clustering_data["plant"] == plant_id
        plant_results = clustering_data[plant_mask].copy()
        
        if not plant_results.empty:
            plant_results = plant_results.drop(columns=["plant"])
            output_file = output_path / f"{plant_id}_clustered.csv"
            plant_results.to_csv(output_file, index=False)
            
            if verbose_output:
                print(f"    Saved {plant_id}: {len(plant_results)} clustered samples")
    
    if verbose_output:
        print("Pollution clustering complete.")


def _create_clustering_plot(features, labels, data, title, output_path=None, filename="clusters_pca.png"):
    """Create PCA visualization of clusters matching the original style"""
    
    # PCA um die Cluster zu visualisieren
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)

    # Visualisierung der Cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title("PCA-Projection of Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plot_file = output_path / filename
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
