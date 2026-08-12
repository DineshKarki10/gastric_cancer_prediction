import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_k_distance(distances, min_samples, output_dir):
    """Generates and saves the K-nearest-neighbor distance curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, color='blue', lw=2)
    plt.title(f'Sorted Distance to {min_samples}-th Nearest Neighbor', fontsize=14)
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'{min_samples}-NN Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight some percentiles
    p90 = np.percentile(distances, 90)
    p50 = np.percentile(distances, 50)
    plt.axhline(y=p90, color='red', linestyle='--', label=f'90th Percentile ({p90:.2f})')
    plt.axhline(y=p50, color='orange', linestyle='--', label=f'Median ({p50:.2f})')
    plt.legend()
    
    plot_path = os.path.join(output_dir, 'k_distance_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved K-distance plot to: {plot_path}")


def plot_clusters(X, labels, method_name, output_dir):
    """Generates a 2D PCA projection of the clusters and saves it."""
    print("Computing PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels
    })
    
    plt.figure(figsize=(10, 8))
    
    # Sort cluster names so noise (-1) is plotted first or distinctively
    unique_labels = np.unique(labels)
    
    # Create color palette
    n_colors = len(unique_labels)
    if -1 in unique_labels:
        colors = sns.color_palette("husl", n_colors - 1)
        palette = {-1: (0.7, 0.7, 0.7)}
        # map other clusters
        j = 0
        for lbl in unique_labels:
            if lbl != -1:
                palette[lbl] = colors[j]
                j += 1
    else:
        palette = "husl"
        
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Cluster', data=df_pca,
        palette=palette, alpha=0.7, edgecolor='none', s=25
    )
    
    plt.title(f'2D PCA Projection of {method_name.upper()} Clusters', fontsize=14)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(output_dir, 'clustering_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved cluster scatter plot to: {plot_path}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def load_data(data_path):
    """Loads the dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    print(f"Loading dataset from: {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def run_k_distance_analysis(X, min_samples, plot_dir=None):
    """
    Computes the distance of each point to its k-th nearest neighbor,
    where k = min_samples. Sorting and printing these distances helps
    identify the 'elbow' point to select an appropriate 'eps' for DBSCAN.
    """
    section("K-NEAREST NEIGHBOR DISTANCE ANALYSIS (FOR EPS TUNING)")
    print(f"Calculating distances to the {min_samples}-th nearest neighbor...")
    start_time = time.time()
    
    # Fit NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Sort the distances (the last column corresponds to the k-th neighbor distance)
    k_distances = np.sort(distances[:, -1])
    duration = time.time() - start_time
    print(f"Distance calculation completed in {duration:.2f} seconds.")
    
    print("\nNearest Neighbor Distance Summary statistics:")
    print(f"  Min Distance: {k_distances.min():.4f}")
    print(f"  Max Distance: {k_distances.max():.4f}")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(k_distances, percentiles)
    print("  Percentiles:")
    for p, val in zip(percentiles, perc_vals):
        print(f"    {p:2d}%: {val:.4f}")
        
    print("\nRecommendations for setting 'eps' parameter:")
    print("  - Setting eps below the 5% percentile (e.g. < 1.3) will likely mark >95% of points as noise.")
    print("  - Setting eps near the 90%-95% percentile (e.g. 1.7 - 1.8) will group most points into one or few clusters.")
    print("  - Look for an 'elbow' or inflection point where the distance curve bends upwards.")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_k_distance(k_distances, min_samples, plot_dir)


def fit_dbscan(X, eps, min_samples):
    """Fits DBSCAN clustering model."""
    section("FITTING DBSCAN MODEL")
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    start_time = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    duration = time.time() - start_time
    print(f"DBSCAN training completed in {duration:.2f} seconds.")
    return labels


def fit_hdbscan(X, min_cluster_size, min_samples):
    """Fits HDBSCAN clustering model."""
    section("FITTING HDBSCAN MODEL")
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    start_time = time.time()
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=-1)
    labels = hdb.fit_predict(X)
    duration = time.time() - start_time
    print(f"HDBSCAN training completed in {duration:.2f} seconds.")
    return labels


def analyze_clusters(df, labels, target_col="label"):
    """Performs cluster composition analysis and logs statistics."""
    section("CLUSTER PROFILE AND QUALITY ANALYSIS")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    total_points = len(labels)
    
    print(f"Number of clusters found: {n_clusters}")
    
    # Calculate noise percentage
    if -1 in unique_labels:
        n_noise = counts[unique_labels == -1][0]
        noise_pct = (n_noise / total_points) * 100
        print(f"Noise points (outliers): {n_noise} ({noise_pct:.2f}%)")
    else:
        print("Noise points (outliers): 0 (0.00%)")
        
    print("\nCluster Distribution:")
    for label, count in zip(unique_labels, counts):
        lbl_str = "Noise (-1)" if label == -1 else f"Cluster {label}"
        pct = (count / total_points) * 100
        print(f"  {lbl_str:12s}: {count:6d} ({pct:.2f}%)")
        
    # Analyze correlation of clusters with target label if present
    if target_col in df.columns:
        print(f"\nGastric Cancer Rate (target='{target_col}') per Cluster:")
        temp_df = pd.DataFrame({"cluster": labels, "target": df[target_col]})
        grouped = temp_df.groupby("cluster")["target"].agg(["count", "mean"])
        grouped.columns = ["Size", "Cancer Rate"]
        grouped.index = [f"Cluster {i}" if i != -1 else "Noise (-1)" for i in grouped.index]
        print(grouped)
        
        # Compute cramer's v or correlation for association
        non_noise_mask = (labels != -1)
        if non_noise_mask.sum() > 0 and n_clusters > 1:
            try:
                # Simple correlation check for non-noise points
                corr = np.corrcoef(labels[non_noise_mask], df[target_col][non_noise_mask].astype(float))[0, 1]
                print(f"\nCorrelation between cluster ID and cancer target (excluding noise): {corr:.4f}")
            except Exception as e:
                pass


def compute_metrics(X, labels):
    """
    Computes silhouette and Davies-Bouldin scores.
    Warning: Silhouette score calculation is O(N^2) memory/time complexity,
    so we compute it on a sample of max 5,000 points if the dataset is large.
    """
    section("CLUSTERING METRICS")
    
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        print("Cannot compute clustering metrics: only 1 cluster/noise group identified.")
        return
        
    # Sample if too large to prevent crash/hang
    max_metric_sample = 5000
    if len(X) > max_metric_sample:
        print(f"Sampling {max_metric_sample} points to calculate clustering metrics...")
        indices = np.random.choice(len(X), max_metric_sample, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
        
    unique_sample_labels = set(labels_sample)
    if len(unique_sample_labels) <= 1:
        print("Metrics sample contains only one cluster. Skipping metrics.")
        return
        
    print("Calculating metrics...")
    try:
        db_score = davies_bouldin_score(X_sample, labels_sample)
        print(f"  Davies-Bouldin Score: {db_score:.4f} (lower is better, represents cluster separation)")
        
        sil_score = silhouette_score(X_sample, labels_sample)
        print(f"  Silhouette Coefficient: {sil_score:.4f} (-1 to 1; higher is better, represents cohesion & separation)")
    except Exception as e:
        print(f"Error calculating metrics: {e}")


def main():
    parser = argparse.ArgumentParser(description="DBSCAN and HDBSCAN Clustering Pipeline")
    parser.add_argument("--data_path", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_cleaned.csv",
                        help="Path to the cleaned dataset CSV")
    parser.add_argument("--output_path", type=str, default="/Users/Sanskar/Documents/gastric_cancer_prediction/Dataset/gastric_cancer_clustered.csv",
                        help="Path to save the output CSV with cluster labels")
    parser.add_argument("--method", type=str, choices=["dbscan", "hdbscan"], default="dbscan",
                        help="Clustering method to use (dbscan or hdbscan)")
    parser.add_argument("--eps", type=float, default=1.8,
                        help="Epsilon parameter for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=18,
                        help="Minimum samples parameter for core points in DBSCAN/HDBSCAN")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size parameter for HDBSCAN")
    parser.add_argument("--sample_size", type=int, default=20000,
                        help="Number of rows to sample for clustering. Set to 0 to run on full dataset")
    parser.add_argument("--k_distance", action="store_true",
                        help="Run K-nearest neighbors distance analysis for tuning eps and exit")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Directory to save generated charts (k-distance plot, 2D projection)")
    
    args = parser.parse_args()
    
    # 1. Load Data
    df = load_data(args.data_path)
    
    # 2. Extract and scale continuous numerical features
    mirna_cols = ["diana_microt", "elmmo", "microcosm", "miranda", "mirdb", "pictar", "pita", "targetscan"]
    continuous_cols = ["age"] + mirna_cols
    
    # If the user has other continuous columns, we can add them here
    for col in ["predicted.sum", "all.sum"]:
        if col in df.columns:
            continuous_cols.append(col)
            
    print(f"Continuous columns selected for clustering: {continuous_cols}")
    
    # Handle subsampling if requested
    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Subsampling dataset to {args.sample_size:,} random rows for analysis/clustering...")
        # Fix seed for reproducibility
        df_model = df.sample(n=args.sample_size, random_state=42).copy()
    else:
        df_model = df.copy()
        
    # Scale variables (crucial since DBSCAN relies on Euclidean distances)
    print("Standardizing numerical features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model[continuous_cols])
    
    # 3. Parameter selection path
    if args.k_distance:
        run_k_distance_analysis(X, args.min_samples, args.plot_dir)
        return
        
    # 4. Clustering Execution
    if args.method == "dbscan":
        labels = fit_dbscan(X, args.eps, args.min_samples)
        lbl_col_name = f"dbscan_cluster"
    else:
        labels = fit_hdbscan(X, args.min_cluster_size, args.min_samples)
        lbl_col_name = f"hdbscan_cluster"
        
    # Append labels to the processed dataframe
    df_model[lbl_col_name] = labels
    
    # 5. Analysis
    analyze_clusters(df_model, labels)
    compute_metrics(X, labels)
    
    # Generate visualization plots if plot_dir is provided
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_clusters(X, labels, args.method, args.plot_dir)
        
    # 6. Save results
    section("SAVING RESULTS")
    # Save the dataframe
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_model.to_csv(args.output_path, index=False)
    print(f"Successfully saved clustered dataset to: {args.output_path}")
    print(f"Dataset shape with cluster labels: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")


if __name__ == "__main__":
    main()
