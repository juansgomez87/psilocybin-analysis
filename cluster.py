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
import random
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in matmul') 
warnings.filterwarnings('ignore', message='divide by zero encountered in matmul')
warnings.filterwarnings('ignore', message='overflow encountered in matmul') 
warnings.filterwarnings('ignore', message='invalid value encountered in divide') 

import pdb

class PsiloClusterer():
    def __init__(self, algo, mean, clustering_method, plot_flag, playlist, label_type, clip=False, boundaries=False):
        self.seed = 1987
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.clustering_method = clustering_method
        self.label_type = label_type

        self.algo = algo
        self.df = pd.read_csv(f'data/df_{algo}{mean}.csv', index_col=0)
        self.plot_flag = plot_flag
        self.playlist = playlist
        self.clip = clip
        self.boundaries = boundaries


        if self.label_type == 'phase' and self.boundaries:
            # always find songs at the boundaries of each phase for whole songs
            temp = pd.read_csv(f'data/df_{algo}_mean.csv', index_col=0)
            temp = temp.reset_index(drop=True)
            # from onset to peak
            boundaries_onset_peak = [(i, i+1) for i in range(len(temp) - 1) if temp.loc[i, 'phase'] == 'onset' and temp.loc[i+1, 'phase'] == 'peak']
            # flatten the list of tuples
            boundaries_onset_peak = [i for t in boundaries_onset_peak for i in t]
            # from peak to return
            boundaries_peak_return = [(i, i+1) for i in range(len(temp) - 1) if temp.loc[i, 'phase'] == 'peak' and temp.loc[i+1, 'phase'] == 'return']
            # flatten the list of tuples
            boundaries_peak_return = [i for t in boundaries_peak_return for i in t]
            # get the index of the songs at the boundaries
            boundaries_idx = boundaries_onset_peak + boundaries_peak_return
            # get spotify ids 
            spotify_ids = temp.loc[boundaries_idx, 'spotify_id'].values
            # drop all rows with that spotify id
            initial_len = len(self.df)
            self.df = self.df[~self.df['spotify_id'].isin(spotify_ids)]
            removed_len = initial_len - len(self.df)
            if mean:
                print(f'Removing {removed_len} songs at the boundaries of each phase for whole songs!')
            else:
                print(f'Removing {removed_len} chunks at the boundaries of each phase for chunks!')


        if self.label_type == 'phase' and self.clip:
            print('Clipping the data of return phase!')
            # use full data to get durations and clip the return phase
            temp = pd.read_csv(f'data/full_data.csv', index_col=0)
            temp = temp[temp['process?'] == True].copy()

            return_durations = [(_, temp[(temp.playlist == _) & (temp.phase == 'return')]['Duration (m)'].sum()) for _ in temp['playlist'].unique()]
            shortest = min(return_durations, key=lambda x: x[1])
            # ('imperial2', np.float64(126.983))
            print(f'Shortest return phase is {shortest[0]} with duration {shortest[1]} minutes!')
            # clip the return phase of all other playlists to the duration of the shortest return phase
            spotify_ids = []
            for playlist in temp['playlist'].unique():
                if playlist != shortest[0]:
                    df_playlist = temp[temp['playlist'] == playlist]
                    df_playlist = df_playlist[df_playlist['phase'] == 'return']
                    df_playlist = df_playlist.reset_index(drop=True)                    
                    # Calculate cumulative sum of durations
                    df_playlist['cumulative_duration'] = df_playlist['Duration (m)'].cumsum()
                    
                    # Find rows that need to be dropped (those that exceed the threshold)
                    rows_to_drop = df_playlist[df_playlist['cumulative_duration'] > shortest[1]]
                    ids = rows_to_drop['link'].str.extract(r'track/([a-zA-Z0-9]+)')[0].tolist()
                    spotify_ids.extend(ids)
                    # print(f'Playlist {playlist}: Dropping {len(ids)} songs with Spotify IDs: {ids}')

            initial_len = len(self.df)
            self.df = self.df[~self.df['spotify_id'].isin(spotify_ids)]
            removed_len = initial_len - len(self.df)
            if mean:
                print(f'Removing {removed_len} songs by clipping return phase!')
            else:
                print(f'Removing {removed_len} chunks by clipping return phase!')

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

        # Define -and label columns
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
        
        unique_true_labels = np.unique(self.y_true)
        unique_pred_labels = np.unique(self.y_pred)
        
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        pred_label_to_num = {label: i for i, label in enumerate(unique_pred_labels)}
        
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        y_pred_num = np.array([pred_label_to_num[label] for label in self.y_pred])
        
        cm = confusion_matrix(y_true_num, y_pred_num)
        
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        cluster_mapping = {}
        for true_idx, pred_idx in zip(row_indices, col_indices):
            pred_label = unique_pred_labels[pred_idx]
            true_label = unique_true_labels[true_idx]
            cluster_mapping[pred_label] = true_label
        
        y_pred_aligned = np.array([cluster_mapping.get(pred, pred) for pred in self.y_pred])
        
        cm_aligned = confusion_matrix(self.y_true, y_pred_aligned)
        
        return cm_aligned, [cluster_mapping.get(unique_pred_labels[i], unique_pred_labels[i]) for i in range(len(unique_pred_labels))]
        
    def show_cluster_mapping(self):
        """Show which predicted cluster corresponds to which true class."""
        from scipy.optimize import linear_sum_assignment
        
        unique_true_labels = np.unique(self.y_true)
        unique_pred_labels = np.unique(self.y_pred)
        
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        pred_label_to_num = {label: i for i, label in enumerate(unique_pred_labels)}
        
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        y_pred_num = np.array([pred_label_to_num[label] for label in self.y_pred])
        
        cm = confusion_matrix(y_true_num, y_pred_num)
        
        cost_matrix = -cm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        cluster_mapping = {}
        for true_idx, pred_idx in zip(row_indices, col_indices):
            pred_label = unique_pred_labels[pred_idx]
            true_label = unique_true_labels[true_idx]
            cluster_mapping[pred_label] = true_label
        
        for pred_cluster in sorted(unique_pred_labels):
            true_class = cluster_mapping.get(pred_cluster, "No clear mapping")
        
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
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(self.X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        unique_true_labels = np.unique(self.y_true)
        true_label_to_num = {label: i for i, label in enumerate(unique_true_labels)}
        y_true_num = np.array([true_label_to_num[label] for label in self.y_true])
        
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true_num, cmap='tab10', alpha=0.7)
        ax1.set_title(f'True {self.label_type} Labels')
        ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_ticks(range(len(unique_true_labels)))
        cbar1.set_ticklabels(unique_true_labels)
        
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=self.y_pred, cmap='tab10', alpha=0.7)
        ax2.set_title(f'{self.clustering_method.upper()} Clustering Results')
        ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        
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
    parser.add_argument('-clip', 
                        dest='clip',
                        action='store_true',
                        default=False,
                        help='Clip the data of return phase.')
    parser.add_argument('-boundaries', 
                        dest='boundaries',
                        action='store_true',
                        default=False,
                        help='Remove ')                        
    args = parser.parse_args()

    if args.boundaries and args.clip:
        print('Cannot clip and remove boundaries at the same time!')
        exit()

    mean_str = '_mean' if args.mean else ''

    clusterer = PsiloClusterer( 
                        algo='compare_lld', 
                        mean=mean_str, 
                        clustering_method=args.method,
                        plot_flag=args.plot, 
                        playlist=args.playlist,
                        label_type=args.label,
                        clip=args.clip,
                        boundaries=args.boundaries)   