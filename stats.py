import pandas as pd
import numpy as np
import argparse
import os
import pdb

import seaborn as sns
import matplotlib.pyplot as plt
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
        # only select features that are means, not standard deviations, or percentiles, or speeds
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
        # Generate distinct colors for each playlist
        n_colors = len(playlists)
        colors = [colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.8) for i in range(n_colors)]
        colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                 for r, g, b in colors]
        playlist_colors = dict(zip(playlists, colors))

        
        pivot = this_df.pivot_table(
            index="playlist",    
            columns="phase",      
            values="Duration (m)", 
            aggfunc="sum",      
            fill_value=0          
        )
        pivot_perc = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # 4. Plot with seaborn
        # Letter size page: 8.5" x 11", single column width ~3.5-4"
        fig, ax = plt.subplots(figsize=(5, 2.5))

        # Horizontal stacked bar
        pivot_perc.plot(
            kind="barh",           # horizontal
            stacked=True,
            ax=ax,
            colormap="Set2"
        )

        # Titles and labels
        # ax.set_title("Phase distribution per playlist", fontsize=9)
        ax.set_xlabel("Duration (%)", fontsize=8)  # horizontal → xlabel
        ax.set_ylabel("Playlist", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        plt.savefig('figs/phase_distribution.pdf', bbox_inches='tight', dpi=300)

        # Select numerical features for analysis
        numerical_features = ['BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 
                            'Happy', 'Speech', 'Live', 'Loud (Db)']
        this_df[numerical_features[1:-1]] /= 100

        # 1. Basic descriptive statistics by playlist
        stats_df = this_df.groupby('playlist')[numerical_features].agg(['mean', 'std']).round(2)
        print("\nDescriptive Statistics by Playlist:")
        print(stats_df)



        # 2. Create wordcloud for genres by playlist (Combined version)
        genre_by_playlist = {}
        for playlist in playlists:
            playlist_genres = this_df[this_df['playlist'] == playlist]['Genres'].str.split(',').explode()
            # playlist_genres = this_df[this_df['playlist'] == playlist]['Parent Genres'].str.split(',').explode()
            genre_counts = playlist_genres.value_counts()
            genre_by_playlist[playlist] = genre_counts
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            for playlist, genres in genre_by_playlist.items():
                if word in genres.index:
                    return playlist_colors[playlist]
            return '#808080'  # default gray for any unmatched genres
        
        all_genres = pd.concat([counts for counts in genre_by_playlist.values()])
        genre_freq = all_genres.groupby(all_genres.index).sum()
        
        # Combined wordcloud
        wordcloud = WordCloud(width=1200, height=800,
                            background_color='white',
                            color_func=color_func,
                            max_words=150,
                            prefer_horizontal=0.7)
        
        wordcloud.generate_from_frequencies(genre_freq)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=playlist, markersize=10)
                         for playlist, color in playlist_colors.items()]
        plt.legend(handles=legend_elements, loc='center left', 
                  bbox_to_anchor=(1, 0.5), title='Playlists')
        
        plt.tight_layout()
        plt.savefig('figs/genre_wordcloud_combined.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # 2b. Create individual wordclouds per playlist
        n_cols = 4
        n_rows = (len(playlists) + n_cols - 1) // n_cols  # ceiling division
        
        # Increase figure size and add more vertical space
        fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        # Create grid with specific spacing
        grid = plt.GridSpec(n_rows, n_cols, hspace=0.1, wspace=0.1)
        
        for idx, (playlist, genres) in enumerate(genre_by_playlist.items()):
            # Create wordcloud for this playlist
            wordcloud = WordCloud(width=400, height=300,
                                background_color='white',
                                color_func=lambda *args, **kwargs: playlist_colors[playlist],
                                max_words=50,
                                prefer_horizontal=0.7)
            
            # Generate from this playlist's frequencies
            wordcloud.generate_from_frequencies(genres)
            
            # Create subplot using GridSpec
            ax = plt.subplot(grid[idx // n_cols, idx % n_cols])
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{playlist}\n({len(genres)} unique genres)', pad=10, fontsize=16)
        
        plt.savefig('figs/genre_wordcloud_individual.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Create violin plots for key features
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(3, 3, i)
            sns.violinplot(data=this_df, x='playlist', y=feature, inner='box')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{feature} Distribution by Playlist')
        plt.tight_layout()
        plt.savefig('figs/spoti_feats_dist.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 4. Correlation matrix of features
        plt.figure(figsize=(10, 8))
        correlation_matrix = this_df[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Musical Features')
        plt.tight_layout()
        plt.savefig('figs/spoti_feats_corr.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 5. Run ANOVA tests for each feature
        print("\nANOVA Test Results:")
        for feature in numerical_features:
            groups = [group[feature].values for name, group in this_df.groupby('playlist')]
            f_stat, p_val = f_oneway(*groups)
            print(f"{feature:15} F-statistic: {f_stat:8.2f}, p-value: {p_val:.2e}")
        
        # 6. Create radar plot for playlist characteristics
        num_feats = numerical_features[1:-1]
        means = this_df.groupby('playlist')[numerical_features].mean()

        # Normalize the features for radar plot
        means_normalized = (means - means.min()) / (means.max() - means.min())
        
        # Radar plot setup
        angles = np.linspace(0, 2*np.pi, len(numerical_features), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        # Create subplots grid
        n_playlists = len(means_normalized.index)
        n_cols = 4
        n_rows = (n_playlists + n_cols - 1) // n_cols  # ceiling division
        
        fig = plt.figure(figsize=(20, 4*n_rows))
        
        for idx, playlist in enumerate(means_normalized.index, 1):
            ax = plt.subplot(n_rows, n_cols, idx, projection='polar')
            
            values = means_normalized.loc[playlist].values
            values = np.concatenate((values, [values[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, color=playlist_colors[playlist])
            ax.fill(angles, values, alpha=0.25, color=playlist_colors[playlist])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(numerical_features, fontsize=14)  # Larger, bold radar labels
            ax.set_ylim(0, 1)
            
            # Customize the plot
            ax.set_title(playlist, pad=20, fontsize=16)  # Larger, bold title
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=13)  # Tick label fontsize
            
            # Add gridlines for better readability
            ax.set_rticks([0.2, 0.4, 0.6, 0.8])
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('figs/spoti_radar_subplots.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # 7. Analyze playlist statistics and shared songs
        # Count songs per playlist
        songs_per_playlist = this_df.groupby('playlist').size()
        print("\nNumber of songs per playlist:")
        print(songs_per_playlist)
        total_duration = this_df.groupby('playlist')['Duration (s)'].sum() / 3600  # convert to hours

        # Create a figure with two subplots
        fig = plt.figure(figsize=(8, 4))

        # Plot number of songs
        songs_per_playlist.plot(kind='bar')
        plt.title('Number of Songs per Playlist')
        plt.ylabel('Number of Songs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figs/playlist_stats.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # Find shared songs across playlists
        shared_songs = this_df.groupby('Spotify Track Id').agg({
            'playlist': lambda x: list(set(x)),
            'Artist': 'first',
            'Song': 'first'
        })
        
        # Filter for songs that appear in multiple playlists
        shared_songs['num_playlists'] = shared_songs['playlist'].apply(len)
        shared_songs = shared_songs[shared_songs['num_playlists'] > 1].sort_values('num_playlists', ascending=False)

        # Print shared songs information
        print("\nSongs shared across playlists:")
        print("================================")
        for idx, row in shared_songs.iterrows():
            print(f"\nSong: {row['Song']}")
            print(f"Artist: {row['Artist']}")
            print(f"Appears in {row['num_playlists']} playlists: {', '.join(row['playlist'])}")

        # Create a visualization of shared songs
        plt.figure(figsize=(8, 4))
        shared_counts = shared_songs['num_playlists'].value_counts().sort_index()
        
        bars = plt.bar(shared_counts.index, shared_counts.values)
        plt.title('Distribution of Shared Songs')
        plt.xlabel('Number of Playlists Sharing the Song')
        plt.ylabel('Number of Songs')
        
        # Add value labels on top of each bar
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
        
        alpha_corrected = 0.05 / num_tests  # Bonferroni correction
        significant_feats = []
        cohen_d_values = []

        for feat in self.feats:
            groups = [group[feat].dropna().values for _, group in self.df.groupby(self.group)]

            # ANOVA test
            stat, p_value = f_oneway(*groups)

            if p_value < alpha_corrected:  # Apply Bonferroni correction
                significant_feats.append((feat, p_value))

                # Compute Cohen's D for each pair of playlists

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
        return mean_diff / pooled_std if pooled_std > 0 else 0  # Avoid division by zero


    def plot_violin_plots(self, features):
        """ Plot violin plots for significant features with smaller text. """
        num_feats = min(len(features), 6)  # Limit number of plots
        fig, axes = plt.subplots(nrows=num_feats, figsize=(8, 1.5 * num_feats), sharex=True)

        if num_feats == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for i, feat in enumerate(features[:num_feats]):
            sns.violinplot(data=self.df, x=self.group, y=feat, inner="quartile", ax=axes[i])
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

        sns.barplot(data=df_cohen_sorted, x="Cohen's D", y="Feature", hue=f"{self.group}1", dodge=False)
        
        # plt.axvline(x=0.2, color='green', linestyle="--", label="Small Effect (0.2)")
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