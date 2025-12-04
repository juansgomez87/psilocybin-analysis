import pandas as pd
import numpy as np
import argparse
import os
import pdb

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import f_oneway
from itertools import combinations
from wordcloud import WordCloud
import colorsys

class PsiloStats():
    def __init__(self, algo, mean, group):
        self.seed = 1987
        self.group = group
        self.df = pd.read_csv(f'data/df_{algo}{mean}.csv', index_col=0)

        self.feats = [col for col in self.df.columns if col not in 
                           ['file', 'chunk', 'playlist', 'phase', 'umap_x', 'umap_y', 'artist', 'song', 'spotify_id']]
        # only select low level features
        if algo == 'compare_lld':
            print('Using all features!')
        else:
            print('Using mean features only!')
            self.feats = [_ for _ in self.feats if _.endswith('_amean')]

        print(f'Total of {len(self.feats)} features to analyze!')

        
        self.compare_spotify_features()

        self.compare_features()

    def compare_spotify_features(self):
        """ Compare spotify features across playlists with statistical analysis and visualizations. """
        playlists = ['chacruna_baldwin', 'chacruna_kelan_thomas2', 'compass_v2',
                    'copenhagen', 'imperial1', 'imperial2', 'jh_classical', 'jh_overtone']

        this_df = pd.read_csv('data/full_data.csv', index_col=0)
        n_colors = len(playlists)
        colors = [colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.8) for i in range(n_colors)]
        colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                 for r, g, b in colors]
        playlist_colors = dict(zip(playlists, colors))

        self.playlist_colors = playlist_colors

        
        pivot = this_df.pivot_table(
            index="playlist",    
            columns="phase",      
            values="Duration (m)", 
            aggfunc="sum",      
            fill_value=0          
        )
        pivot_perc = pivot.div(pivot.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))

        plot = pivot.plot(
            kind="barh",           
            stacked=True,
            ax=ax,
            colormap="Set2"
        )
        patches = ax.patches
        n_phases = len(pivot.columns)
        for i, playlist in enumerate(pivot.index):
            base_color = mcolors.hex2color(playlist_colors[playlist])
            for j, phase in enumerate(pivot.columns):
                patch_idx = i * n_phases + j
                if patch_idx < len(patches):
                    shade_factor = 0.6 + (j / n_phases) * 0.4
                    shaded_color = tuple(c * shade_factor for c in base_color)
                    patches[patch_idx].set_facecolor(shaded_color)

        ax.set_xlabel("Absolute Duration (m)", fontsize=8) 
        ax.set_ylabel("Playlist", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        plt.savefig('figs/phase_distribution.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        numerical_features = ['BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 
                            'Happy', 'Speech', 'Live', 'Loud (Db)']
        this_df[numerical_features[1:-1]] /= 100

        # 1. Basic descriptive statistics by playlist
        stats_df = this_df.groupby('playlist')[numerical_features].agg(['mean', 'std', 'max', 'min']).round(2)
        print("\nDescriptive Statistics by Playlist:")
        print(stats_df)
        
        phase_totals = pivot.sum(axis=0)  
        phase_means = pivot.mean(axis=0)   
        phase_mins = pivot.min(axis=0)     
        phase_maxs = pivot.max(axis=0)     
        
        phase_summary_parts = []
        for phase in pivot.columns:
            total = phase_totals[phase]
            mean = phase_means[phase]
            min_val = phase_mins[phase]
            max_val = phase_maxs[phase]
            phase_summary_parts.append(
                f"{total:.1f} minutes in the {phase} phase (mean per playlist = {mean:.1f}, range = {min_val:.1f}-{max_val:.1f})"
            )
        
        phase_summary = "corresponding to " + ", ".join(phase_summary_parts) + "."
        print(f"\n{phase_summary}")


        # 2. Create wordcloud for genres by playlist (Combined version)
        genre_by_playlist = {}
        for playlist in playlists:
            playlist_genres = this_df[this_df['playlist'] == playlist]['Genres'].str.split(',').explode()
            playlist_genres = playlist_genres.str.strip().dropna()
            playlist_genres = playlist_genres[playlist_genres != '']
            genre_counts = playlist_genres.value_counts()
            assert len(genre_counts.index) == len(genre_counts.index.unique()), \
                f"Duplicate genres found in playlist {playlist}"
            genre_by_playlist[playlist] = genre_counts
        
        # 2b. Create individual wordclouds per playlist
        n_cols = 4
        n_rows = (len(playlists) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        grid = plt.GridSpec(n_rows, n_cols, hspace=0.1, wspace=0.1)
        
        for idx, (playlist, genres) in enumerate(genre_by_playlist.items()):
            wordcloud = WordCloud(width=400, height=300,
                                background_color='white',
                                color_func=lambda *args, **kwargs: playlist_colors[playlist],
                                max_words=50,
                                prefer_horizontal=0.7)
            
            wordcloud.generate_from_frequencies(genres)
            
            ax = plt.subplot(grid[idx // n_cols, idx % n_cols])
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            playlist_escaped = playlist.replace('_', r'\_')
            ax.set_title(f"$\\bf{{{playlist_escaped}}}$\n({len(genres)} unique genres)", pad=10, fontsize=16)
        
        plt.savefig('figs/genre_wordcloud_individual.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print("\nANOVA Test Results:")
        for feature in numerical_features:
            groups = [group[feature].values for name, group in this_df.groupby('playlist')]
            f_stat, p_val = f_oneway(*groups)
            print(f"{feature:15} F-statistic: {f_stat:8.2f}, p-value: {p_val:.2e}")
        
        # 6. Create radar plot for playlist characteristics
        num_feats = numerical_features[1:-1]
        means = this_df.groupby('playlist')[numerical_features].mean()
        means_normalized = (means - means.min()) / (means.max() - means.min())
        means_normalized = means.copy()
        for col in ['Loud (Db)', 'BPM']:
            means_normalized[col] = (means[col] - means[col].min()) / (means[col].max() - means[col].min())
        
        angles = np.linspace(0, 2*np.pi, len(numerical_features), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  
        
        n_playlists = len(means_normalized.index)
        n_cols = 4
        n_rows = (n_playlists + n_cols - 1) // n_cols  
        
        fig = plt.figure(figsize=(20, 4*n_rows))
        
        for idx, playlist in enumerate(means_normalized.index, 1):
            ax = plt.subplot(n_rows, n_cols, idx, projection='polar')
            
            values = means_normalized.loc[playlist].values
            values = np.concatenate((values, [values[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, color=playlist_colors[playlist])
            ax.fill(angles, values, alpha=0.25, color=playlist_colors[playlist])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(numerical_features, fontsize=14)  
            ax.set_ylim(0, 1)
            
            playlist_escaped = playlist.replace('_', r'\_')
            ax.set_title(f"$\\bf{{{playlist_escaped}}}$", pad=20, fontsize=16)  
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=13) 
            
            ax.set_rticks([0.2, 0.4, 0.6, 0.8])
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('figs/spoti_radar_subplots.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # 7. Analyze playlist statistics and shared songs
        songs_per_playlist = this_df.groupby('playlist').size()
        print("\nNumber of songs per playlist:")
        print(songs_per_playlist)
        total_duration = this_df.groupby('playlist')['Duration (s)'].sum() / 3600  # convert to hours

        fig = plt.figure(figsize=(8, 4))

        ax = songs_per_playlist.plot(kind='bar', color=[playlist_colors.get(p, '#333333') for p in songs_per_playlist.index])
        plt.title('Number of Songs per Playlist')
        plt.ylabel('Number of Songs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figs/playlist_stats.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        shared_songs = this_df.groupby('Spotify Track Id').agg({
            'playlist': lambda x: list(set(x)),
            'Artist': 'first',
            'Song': 'first'
        })
        
        shared_songs['num_playlists'] = shared_songs['playlist'].apply(len)
        shared_songs = shared_songs[shared_songs['num_playlists'] > 1].sort_values('num_playlists', ascending=False)

        print("\nSongs shared across playlists:")
        print("================================")
        for idx, row in shared_songs.iterrows():
            print(f"\nSong: {row['Song']}")
            print(f"Artist: {row['Artist']}")
            print(f"Appears in {row['num_playlists']} playlists: {', '.join(row['playlist'])}")

        plt.figure(figsize=(8, 4))
        shared_counts = shared_songs['num_playlists'].value_counts().sort_index()
        
        bars = plt.bar(shared_counts.index, shared_counts.values)
        plt.title('Distribution of Shared Songs')
        plt.xlabel('Number of Playlists Sharing the Song')
        plt.ylabel('Number of Songs')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('figs/shared_songs_dist.pdf', bbox_inches='tight', dpi=300)
        plt.close()


    def compare_features(self):
        """ Run ANOVA with Bonferroni correction and Cohen's D analysis. """
        num_tests = len(self.feats)
        
        alpha_corrected = 0.05 / num_tests  
        significant_feats = []
        cohen_d_values = []

        for feat in self.feats:
            groups = [group[feat].dropna().values for _, group in self.df.groupby(self.group)]

            stat, p_value = f_oneway(*groups)

            if p_value < alpha_corrected:  
                significant_feats.append((feat, p_value))

                playlist_names = self.df[self.group].unique()
                for (p1, p2) in combinations(playlist_names, 2):
                    group1 = self.df[self.df[self.group] == p1][feat].dropna()
                    group2 = self.df[self.df[self.group] == p2][feat].dropna()
                    
                    d_value = self.cohens_d(group1, group2)
                    cohen_d_values.append({"Feature": feat, f"{self.group}1": p1, f"{self.group}2": p2, "Cohen's D": d_value})

        print(f"Significant Features (Bonferroni corrected): {len(significant_feats)} / {num_tests}")
        for feat, p in significant_feats:
            print(f"{feat}: p={p:.4e}")

        if significant_feats:
            self.plot_violin_plots([feat for feat, _ in significant_feats])
            self.plot_cohens_d(pd.DataFrame(cohen_d_values))


    def cohens_d(self, group1, group2):
        """ Compute Cohen's D effect size. """
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                              (len(group2) - 1) * np.var(group2, ddof=1)) /
                             (len(group1) + len(group2) - 2))
        return mean_diff / pooled_std if pooled_std > 0 else 0  


    def plot_violin_plots(self, features):
        """ Plot violin plots for significant features with smaller text. """
        num_feats = min(len(features), 6)  
        fig, axes = plt.subplots(nrows=num_feats, figsize=(8, 1.5 * num_feats), sharex=True)

        if num_feats == 1:
            axes = [axes]  

        palette = None
        if self.group == 'playlist' and hasattr(self, 'playlist_colors'):
            unique_groups = sorted(self.df[self.group].unique())
            palette = [self.playlist_colors.get(g, '#333333') for g in unique_groups]

        for i, feat in enumerate(features[:num_feats]):
            sns.violinplot(data=self.df, x=self.group, y=feat, inner="quartile", ax=axes[i], palette=palette)
            axes[i].set_title(f"Distribution of {feat} Across {self.group}", fontsize=10)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, fontsize=8)
            axes[i].set_ylabel("", fontsize=8)

        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=45, fontsize=8)
        plt.tight_layout()
        plt.show()


    def plot_cohens_d(self, df_cohen):
        """ Plot Cohen's D values as a bar plot. """
        plt.figure(figsize=(12, 6))
        df_cohen = df_cohen[np.abs(df_cohen["Cohen's D"]) >= 0.8]
        df_cohen_sorted = df_cohen.sort_values(by="Cohen's D", ascending=False)

        palette = None
        if self.group == 'playlist' and hasattr(self, 'playlist_colors'):
            hue_col = f"{self.group}1"
            if hue_col in df_cohen_sorted.columns:
                unique_hues = sorted(df_cohen_sorted[hue_col].unique())
                palette = [self.playlist_colors.get(g, '#333333') for g in unique_hues]

        sns.barplot(data=df_cohen_sorted, x="Cohen's D", y="Feature", hue=f"{self.group}1", dodge=False, palette=palette)
        
        plt.axvline(x=0.5, color='orange', linestyle="--", label="Medium Effect (0.5)")
        plt.axvline(x=0.8, color='red', linestyle="--", label="Large Effect (0.8)")

        plt.legend()
        plt.title("Cohen's D Effect Sizes for Significant Features")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistics and plot psilocybin playlists")
    parser.add_argument('-clf', 
                        type=str,
                        choices=['playlist', 'phase'],
                        default='playlist',
                        help='Select labels to choose for classification.')
    args = parser.parse_args()

    mean_str = '' # always calculate stats on features extracted every 30s

    clf = PsiloStats(algo='compare_lld', mean=mean_str, group=args.clf)