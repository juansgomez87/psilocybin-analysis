import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in matmul') 
warnings.filterwarnings('ignore', message='divide by zero encountered in matmul')
warnings.filterwarnings('ignore', message='overflow encountered in matmul') 
warnings.filterwarnings('ignore', message='invalid value encountered in divide') 

import pdb

class PsiloClusterer():
    def __init__(self, algo, mean, clustering_method, plot_flag, playlist, label_type):
        self.seed = 1987
        self.clustering_method = clustering_method
        self.label_type = label_type

        self.algo = algo
        self.df = pd.read_csv(f'data/df_{algo}{mean}.csv', index_col=0)
        self.plot_flag = plot_flag
        self.playlist = playlist

        if playlist == 'all':
            print('Using all playlists!')
            all_playlists = self.df['playlist'].unique().tolist()
            self.df = self.df[self.df['playlist'].isin(all_playlists)]
        elif playlist == 'most':
            # use all playlists except imperial1
            print('Using all playlists except imperial1!')
            all_playlists = [_ for _ in self.df['playlist'].unique().tolist() if _ != 'imperial1']
            self.df = self.df[self.df['playlist'].isin(all_playlists)]
        else:
            self.df = self.df[self.df['playlist'] == playlist]
            print(f'Using only {playlist} playlist!')

        # Define feature and label columns
        self.feature_columns = [col for col in self.df.columns if col not in 
                           ['file', 'chunk', 'phase', 'playlist', 'umap_x', 'umap_y', 'artist', 'song', 'spotify_id']]
        
        # Always do PCA tp 99% of variance
        self.pca_reduction()

        # Get true labels for evaluation
        self.y_true = self.df[label_type].values
        self.unique_labels = np.unique(self.y_true)
        self.n_clusters = len(self.unique_labels)
        
        print(f'True labels: {self.unique_labels}')
        print(f'Number of true clusters: {self.n_clusters}')
        
        # Run clustering
        self.run_clustering()
        
    def pca_reduction(self):
        print('Calculating PCA, this might take time...')
        self.X = self.df[self.feature_columns].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
    
        self.pca = PCA(n_components=0.99)
        X_pca = self.pca.fit_transform(self.X)
        n_comp = X_pca.shape[1]
        print('-'*50)
        print(f'PCA reduced from {len(self.feature_columns)} to {n_comp} to keep 99% of variance!')

        self.feature_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=self.feature_columns, index=self.df.index)
        self.df = pd.concat([self.df, df_pca], axis=1)
        self.X = self.df[self.feature_columns].values


    def run_clustering(self):
        """Run the selected clustering algorithm and evaluate results."""
        print(f'Running {self.clustering_method} clustering...')
        
        if self.clustering_method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        elif self.clustering_method == 'gmm':
            self.clusterer = GaussianMixture(n_components=self.n_clusters, random_state=self.seed)
        elif self.clustering_method == 'agglomerative':
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Fit clustering
        self.y_pred = self.clusterer.fit_predict(self.X)
        
        # Handle DBSCAN noise points (-1 labels)
        if self.clustering_method == 'dbscan':
            n_noise = list(self.y_pred).count(-1)
            print(f'DBSCAN found {n_noise} noise points')
            if n_noise > 0:
                # Relabel noise points to the largest cluster
                unique_pred, counts = np.unique(self.y_pred[self.y_pred != -1], return_counts=True)
                largest_cluster = unique_pred[np.argmax(counts)]
                self.y_pred[self.y_pred == -1] = largest_cluster
        
        # Evaluate clustering
        self.evaluate_clustering()
        
        # Generate visualizations
        if self.plot_flag:
            self.plot_clustering_results()
            self.plot_silhouette_analysis()
            self.plot_confusion_matrix()
        
    def evaluate_clustering(self):
        """Evaluate clustering performance against true labels."""
        print('-'*60)
        print(f'CLUSTERING EVALUATION: {self.clustering_method.upper()}')
        print('-'*60)
        
        # Internal clustering metrics (don't require true labels)
        silhouette_avg = silhouette_score(self.X, self.y_pred)
        print(f'Silhouette Score: {silhouette_avg:.3f}')
        
        # External clustering metrics (compare with true labels)
        ari = adjusted_rand_score(self.y_true, self.y_pred)
        nmi = normalized_mutual_info_score(self.y_true, self.y_pred)
        homogeneity = homogeneity_score(self.y_true, self.y_pred)
        completeness = completeness_score(self.y_true, self.y_pred)
        v_measure = v_measure_score(self.y_true, self.y_pred)
        
        print(f'Adjusted Rand Index: {ari:.3f}')
        print(f'Normalized Mutual Information: {nmi:.3f}')
        print(f'Homogeneity: {homogeneity:.3f}')
        print(f'Completeness: {completeness:.3f}')
        print(f'V-measure: {v_measure:.3f}')
        
        # Cluster distribution
        print('\nCluster Distribution:')
        pred_counts = Counter(self.y_pred)
        true_counts = Counter(self.y_true)
        
        print('Predicted clusters:', dict(pred_counts))
        print('True clusters:', dict(true_counts))
        
        # Confusion matrix with proper label alignment
        print('\nConfusion Matrix:')
        cm, aligned_pred_labels = self.align_cluster_labels()
        print(cm)
        
        # Show label mapping
        print('\nCluster Label Mapping:')
        self.show_cluster_mapping()
        
        # Classification report (treating clustering as classification)
        print('\nClassification Report (clusters as classes):')
        # Use aligned labels for classification report
        _, aligned_pred_labels = self.align_cluster_labels()
        y_pred_aligned = np.array([aligned_pred_labels[list(np.unique(self.y_pred)).index(pred)] for pred in self.y_pred])
        print(classification_report(self.y_true, y_pred_aligned))
        
        # Store metrics for comparison
        self.metrics = {
            'silhouette': silhouette_avg,
            'ari': ari,
            'nmi': nmi,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure
        }
        
        # Compute chance baselines
        self.compute_chance_baselines()
        
    def align_cluster_labels(self):
        """Align cluster labels to minimize confusion matrix off-diagonal elements."""
        from scipy.optimize import linear_sum_assignment
        
        # Convert labels to numeric for confusion matrix computation
        unique_true_labels = np.unique(self.y_true)
        unique_pred_labels = np.unique(self.y_pred)
        
        # Create label encoders
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        pred_label_to_num = {label: i for i, label in enumerate(unique_pred_labels)}
        
        # Convert to numeric
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        y_pred_num = np.array([pred_label_to_num[label] for label in self.y_pred])
        
        # Create confusion matrix with numeric labels
        cm = confusion_matrix(y_true_num, y_pred_num)
        
        # Use Hungarian algorithm to find optimal assignment
        # We want to maximize the sum of diagonal elements
        # So we negate the matrix for minimization
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping from predicted cluster to true label
        cluster_mapping = {}
        for true_idx, pred_idx in zip(row_indices, col_indices):
            pred_label = unique_pred_labels[pred_idx]
            true_label = unique_true_labels[true_idx]
            cluster_mapping[pred_label] = true_label
        
        # Apply mapping to get aligned predictions (using true label names)
        y_pred_aligned = np.array([cluster_mapping.get(pred, pred) for pred in self.y_pred])
        
        # Recompute confusion matrix with aligned labels
        cm_aligned = confusion_matrix(self.y_true, y_pred_aligned)
        
        return cm_aligned, [cluster_mapping.get(unique_pred_labels[i], unique_pred_labels[i]) for i in range(len(unique_pred_labels))]
        
    def show_cluster_mapping(self):
        """Show which predicted cluster corresponds to which true class."""
        from scipy.optimize import linear_sum_assignment
        
        # Convert labels to numeric for confusion matrix computation
        unique_true_labels = np.unique(self.y_true)
        unique_pred_labels = np.unique(self.y_pred)
        
        # Create label encoders
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        pred_label_to_num = {label: i for i, label in enumerate(unique_pred_labels)}
        
        # Convert to numeric
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        y_pred_num = np.array([pred_label_to_num[label] for label in self.y_pred])
        
        # Create confusion matrix with numeric labels
        cm = confusion_matrix(y_true_num, y_pred_num)
        
        # Use Hungarian algorithm to find optimal assignment
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping from predicted cluster to true label
        cluster_mapping = {}
        for true_idx, pred_idx in zip(row_indices, col_indices):
            pred_label = unique_pred_labels[pred_idx]
            true_label = unique_true_labels[true_idx]
            cluster_mapping[pred_label] = true_label
        
        # Show the mapping
        # print('  Predicted Cluster → True Class:')
        for pred_cluster in sorted(unique_pred_labels):
            true_class = cluster_mapping.get(pred_cluster, "No clear mapping")
            # print(f'    Cluster {pred_cluster} → "{true_class}"')
        
        # Show cluster sizes and purity
        # print('\n  Cluster Analysis:')
        for pred_cluster in sorted(unique_pred_labels):
            cluster_mask = self.y_pred == pred_cluster
            cluster_size = np.sum(cluster_mask)
            cluster_true_labels = self.y_true[cluster_mask]
            
            # Find most common true label in this cluster
            true_label_counts = Counter(cluster_true_labels)
            most_common_true = true_label_counts.most_common(1)[0]
            purity = most_common_true[1] / cluster_size
            
            # print(f'    Cluster {pred_cluster}: {cluster_size} samples, {purity:.1%} purity (mostly "{most_common_true[0]}")')
        
    def compute_chance_baselines(self):
        """Compute various chance baselines for clustering evaluation."""
        # print('\n' + '='*60)
        # print('CHANCE BASELINES COMPARISON')
        # print('='*60)
        
        # Get the actual number of clusters found by the algorithm
        n_pred_clusters = len(np.unique(self.y_pred))
        # print(f'True classes: {self.n_clusters}, Predicted clusters: {n_pred_clusters}')
        
        # 1. Random Assignment Baseline - use same number of clusters as predicted
        np.random.seed(self.seed)
        y_random = np.random.randint(0, n_pred_clusters, size=len(self.y_true))
        ari_random = adjusted_rand_score(self.y_true, y_random)
        nmi_random = normalized_mutual_info_score(self.y_true, y_random)
        
        # print(f'Random Assignment Baseline ({n_pred_clusters} clusters):')
        # print(f'  ARI: {ari_random:.3f}')
        # print(f'  NMI: {nmi_random:.3f}')
        
        # 2. Single Cluster Baseline (all samples in one cluster)
        y_single = np.zeros(len(self.y_true), dtype=int)
        ari_single = adjusted_rand_score(self.y_true, y_single)
        nmi_single = normalized_mutual_info_score(self.y_true, y_single)
        
        # print(f'Single Cluster Baseline:')
        # print(f'  ARI: {ari_single:.3f}')
        # print(f'  NMI: {nmi_single:.3f}')
        
        # 3. Perfect Clustering Baseline (true labels as clusters)
        ari_perfect = adjusted_rand_score(self.y_true, self.y_true)
        nmi_perfect = normalized_mutual_info_score(self.y_true, self.y_true)
        
        # print(f'Perfect Clustering Baseline:')
        # print(f'  ARI: {ari_perfect:.3f}')
        # print(f'  NMI: {nmi_perfect:.3f}')
        
        # 4. Permutation Baseline (shuffle true labels)
        y_permuted = np.random.permutation(self.y_true)
        ari_permuted = adjusted_rand_score(self.y_true, y_permuted)
        nmi_permuted = normalized_mutual_info_score(self.y_true, y_permuted)
        
        print(f'Permutation Baseline:')
        print(f'  ARI: {ari_permuted:.3f}')
        print(f'  NMI: {nmi_permuted:.3f}')
        
        # 5. Theoretical Random Baseline
        # For ARI: expected value is 0 for random clustering
        # For NMI: expected value is 0 for random clustering
        expected_ari = 0.0
        expected_nmi = 0.0
        
        # print(f'Theoretical Random Baseline:')
        # print(f'  Expected ARI: {expected_ari:.3f}')
        # print(f'  Expected NMI: {expected_nmi:.3f}')
        
        # 6. Random with True Number of Clusters
        y_random_true = np.random.randint(0, self.n_clusters, size=len(self.y_true))
        ari_random_true = adjusted_rand_score(self.y_true, y_random_true)
        nmi_random_true = normalized_mutual_info_score(self.y_true, y_random_true)
        
        # print(f'Random Assignment Baseline ({self.n_clusters} clusters):')
        # print(f'  ARI: {ari_random_true:.3f}')
        # print(f'  NMI: {nmi_random_true:.3f}')
        
        # Compare with actual results
        print(f'\n{self.clustering_method.upper()} vs Baselines:')
        print(f'  ARI: {self.metrics["ari"]:.3f} (vs random: {ari_random:.3f}, vs perfect: {ari_perfect:.3f})')
        print(f'  NMI: {self.metrics["nmi"]:.3f} (vs random: {nmi_random:.3f}, vs perfect: {nmi_perfect:.3f})')
        
        # Statistical significance assessment
        baseline_aris = [ari_random, ari_single, ari_permuted, ari_random_true]
        baseline_nmis = [nmi_random, nmi_single, nmi_permuted, nmi_random_true]
        
        if self.metrics["ari"] > max(baseline_aris):
            print(f'✅ {self.clustering_method.upper()} significantly outperforms all baselines!')
        elif self.metrics["ari"] > max(ari_random, ari_random_true):
            print(f'✅ {self.clustering_method.upper()} outperforms random assignment')
        else:
            print(f'❌ {self.clustering_method.upper()} does not outperform random assignment')
            
        # Store baseline metrics
        self.baseline_metrics = {
            'random_ari': ari_random,
            'random_nmi': nmi_random,
            'random_true_ari': ari_random_true,
            'random_true_nmi': nmi_random_true,
            'single_ari': ari_single,
            'single_nmi': nmi_single,
            'perfect_ari': ari_perfect,
            'perfect_nmi': nmi_perfect,
            'permuted_ari': ari_permuted,
            'permuted_nmi': nmi_permuted
        }
        
    def plot_clustering_results(self):
        """Plot clustering results in 2D PCA space."""
        # Project to 2D for visualization
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(self.X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert string labels to numeric for plotting
        unique_true_labels = np.unique(self.y_true)
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        
        # True labels
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true_num, cmap='tab10', alpha=0.7)
        ax1.set_title(f'True {self.label_type} Labels')
        ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        
        # Create custom colorbar labels
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_ticks(range(len(unique_true_labels)))
        cbar1.set_ticklabels(unique_true_labels)
        
        # Predicted clusters
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=self.y_pred, cmap='tab10', alpha=0.7)
        ax2.set_title(f'{self.clustering_method.upper()} Clustering Results')
        ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        
        # Create custom colorbar labels for predicted clusters
        unique_pred_labels = np.unique(self.y_pred)
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_ticks(range(len(unique_pred_labels)))
        cbar2.set_ticklabels([f'Cluster {i}' for i in unique_pred_labels])
        
        plt.tight_layout()
        plt.savefig(f'figs/{self.label_type}_{self.algo}_{self.clustering_method}_clustering_2d.pdf')
        plt.show()
        
    def plot_silhouette_analysis(self):
        """Plot silhouette analysis for clustering."""
        from sklearn.metrics import silhouette_samples
        
        silhouette_vals = silhouette_samples(self.X, self.y_pred)
        y_lower = 10
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(len(np.unique(self.y_pred))):
            cluster_silhouette_vals = silhouette_vals[self.y_pred == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(i / len(np.unique(self.y_pred)))
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.axvline(x=self.metrics['silhouette'], color="red", linestyle="--", 
                  label=f'Silhouette Score: {self.metrics["silhouette"]:.3f}')
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        ax.set_title(f'Silhouette Analysis for {self.clustering_method.upper()} Clustering')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'figs/{self.label_type}_{self.algo}_{self.clustering_method}_silhouette.pdf')
        plt.show()
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix with proper label alignment."""
        cm, aligned_pred_labels = self.align_cluster_labels()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create display labels
        true_labels = [str(label) for label in self.unique_labels]
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=true_labels)
        disp.plot(
            ax=ax,
            xticks_rotation=45,
            cmap='Blues',
            include_values=True,
            colorbar=False,
            values_format='d'
        )
        
        # Customize the plot
        ax.set_xlabel('Predicted Cluster (Aligned)', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix: {self.clustering_method.upper()} vs {self.label_type.title()}', fontsize=14)
        
        # Add cluster mapping information
        mapping_text = "Cluster Mapping:\n"
        unique_pred_labels = np.unique(self.y_pred)
        for pred_cluster in sorted(unique_pred_labels):
            if pred_cluster in aligned_pred_labels:
                true_class = aligned_pred_labels[list(unique_pred_labels).index(pred_cluster)]
                mapping_text += f"Cluster {pred_cluster} → {true_class}\n"
        
        ax.text(1.02, 0.5, mapping_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'figs/{self.label_type}_{self.algo}_{self.clustering_method}_confusion_matrix.pdf')
        plt.show()
        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster psilocybin music features")
    parser.add_argument('-label', 
                        type=str,
                        choices=['playlist', 'phase'],
                        default='phase',
                        help='Select labels to compare clustering against.')
    parser.add_argument('-mean', 
                        dest='mean',
                        action='store_true',
                        default=False,
                        help='Select mean (song-level) features for clustering.')
    parser.add_argument('-method', 
                        type=str,
                        choices=['kmeans', 'gmm', 'agglomerative'],
                        default='kmeans',
                        help='Select clustering algorithm.')
    parser.add_argument('-plot', 
                        dest='plot',
                        action='store_true',
                        default=False,
                        help='Select to create plots.')
    parser.add_argument('-playlist', 
                        type=str,
                        choices=['all', 'chacruna_baldwin', 'chacruna_kelan_thomas2', 'compass_v2',
                                 'copenhagen', 'imperial1', 'imperial2', 'jh_classical', 'jh_overtone', 'most'],
                        default='all',
                        help='Select the playlist you want to analyze.')
    args = parser.parse_args()

    mean_str = '_mean' if args.mean else ''

    clusterer = PsiloClusterer( 
                        algo='compare_lld', 
                        mean=mean_str, 
                        clustering_method=args.method,
                        plot_flag=args.plot, 
                        playlist=args.playlist,
                        label_type=args.label)   