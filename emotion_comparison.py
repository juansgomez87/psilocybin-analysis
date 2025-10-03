import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from tqdm import tqdm
import seaborn as sns
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pdb



def extract_order_spotify_id(full_path):
    playlist = full_path.split('/')[6]
    filename = full_path.split('/')[-1]

    track_order = int(filename.split('-')[0])
    spotify_id = filename.split('-')[1]
    return track_order, spotify_id, playlist


def count_quadrants(df, x_col, y_col):
    q1 = ((df[x_col] >= 0.0) & (df[y_col] >= 0.0)).sum()  # Top-right
    q2 = ((df[x_col] < 0.0) & (df[y_col] >= 0.0)).sum()   # Top-left
    q3 = ((df[x_col] < 0.0) & (df[y_col] < 0.0)).sum()    # Bottom-left
    q4 = ((df[x_col] >= 0.0) & (df[y_col] < 0.0)).sum()   # Bottom-right
    
    total = df.shape[0]  # Total points
    return {'Top-Right (Q1)': q1, 'Top-Left (Q2)': q2, 'Bottom-Left (Q3)': q3, 'Bottom-Right (Q4)': q4, 'Total': total}

def get_emotions():
    emo_csv = 'data/music_2_emo.csv'
    if os.path.exists(emo_csv):
        emo_df = pd.read_csv(emo_csv)
    else:
        from Music2Emotion.music2emo import Music2emo
        music2emo = Music2emo()

        all_songs = glob.glob('/Users/juangomez/Data/psilocybin/audio/*/*.mp3')
        songs_data = {}
        for song in tqdm(all_songs):
            res = music2emo.predict(song)

            valence = res["valence"]
            arousal = res["arousal"]
            predicted_moods = res["predicted_moods"]

            songs_data[song] = {
                'moods': predicted_moods,
                'arousal': arousal,
                'valence': valence
            }
        emo_df = pd.DataFrame.from_dict(songs_data, orient='index').reset_index()
        emo_df.columns = ['song', 'moods', 'arousal', 'valence']

        emo_df['playlist'] = [_.split('/')[6] for _ in emo_df.song]
        emo_df.to_csv('data/music_2_emo.csv')
    return emo_df

# def plot_arousal_valence(df, emo_df, color='playlist'):
#     colors = {p: c for p, c in zip(df[color].unique(), sns.color_palette('tab10', n_colors=df[color].nunique()))}
#     fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

#     emo_df.dropna(inplace=True)
#     # Plot for Spotify data
#     scatter1 = axes[0].scatter(
#         df['valence'], 
#         df['arousal'], 
#         c=df[color].map(colors), 
#         label=f'Spotify {df.shape[0]} songs',
#         s=5
#     )
#     axes[0].set_xlabel('Valence')
#     axes[0].set_ylabel('Arousal')
#     axes[0].set_xlim([-1, 1])
#     axes[0].set_ylim([-1, 1])
#     axes[0].set_title('Spotify Data')
#     axes[0].axhline(y=0.0, color='black', linestyle='--', linewidth=1)
#     axes[0].axvline(x=0.0, color='black', linestyle='--', linewidth=1)

#     # Plot for Music2Emo data
#     scatter2 = axes[1].scatter(
#         emo_df['valence'], 
#         emo_df['arousal'], 
#         label=f'Mus2Emo {emo_df.shape[0]} songs', 
#         c=emo_df[color].map(colors),
#         s=5
#     )
#     axes[1].set_xlabel('Valence')
#     axes[1].set_xlim([-1, 1])
#     axes[1].set_ylim([-1, 1])
#     axes[1].set_title('Mus2Emo Data')
#     axes[1].axhline(y=0.0, color='black', linestyle='--', linewidth=1)
#     axes[1].axvline(x=0.0, color='black', linestyle='--', linewidth=1)

#     handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=p) 
#             for p, c in colors.items()]

#     fig.subplots_adjust(right=0.85) 
#     legend = plt.legend(handles=handles, title=color, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout(rect=[0, 0, 0.95, 1])
#     plt.savefig(f'figs/playlists_AV_{color}.pdf', bbox_inches='tight', dpi=300)
#     plt.close()

def plot_arousal_valence_compact(df, emo_df, color='playlist'):
    """Compact version of plot_arousal_valence with consistent legend box size."""
    # Ensure both dataframes use the same color mapping
    all_categories = df[color].unique()
    colors = {p: c for p, c in zip(all_categories, sns.color_palette('tab10', n_colors=len(all_categories)))}
    
    # Create figure with fixed proportions
    fig = plt.figure(figsize=(15, 5))  # Wider figure to accommodate fixed legend
    
    # Create axes with absolute positioning
    ax1 = fig.add_axes([0.08, 0.12, 0.35, 0.8])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.48, 0.12, 0.35, 0.8])
    legend_ax = fig.add_axes([0.88, 0.12, 0.1, 0.8])  # Fixed width for legend
    legend_ax.axis('off')
    
    # Plot Spotify data
    ax1.scatter(df['valence'], df['arousal'], c=df[color].map(colors), s=5)
    ax1.set(xlabel='Valence', ylabel='Arousal', xlim=(-1, 1), ylim=(-1, 1), title='Spotify Data')
    ax1.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(x=0.0, color='black', linestyle='--', linewidth=1)
    
    # Plot Music2Emo data
    emo_df_clean = emo_df.dropna()
    ax2.scatter(emo_df_clean['valence'], emo_df_clean['arousal'], 
               c=emo_df_clean[color].map(colors), s=5)
    ax2.set(xlabel='Valence', xlim=(-1, 1), ylim=(-1, 1), title='Mus2Emo Data')
    ax2.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=0.0, color='black', linestyle='--', linewidth=1)
    
    # Create legend with fixed size
    max_length = 22  # Length of 'chacruna_kelan_thomas2'
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, 
                         markersize=8, label=f"{p:<{max_length}}") for p, c in colors.items()]
    
    # Add dummy handles to ensure consistent height (8 items total)
    while len(handles) < 8:
        dummy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w', 
                          markersize=0, label=' ' * max_length)
        handles.append(dummy)
    
    # Place legend with fixed position and size
    legend = legend_ax.legend(handles=handles, 
                            title=color,
                            loc='center left',
                            borderpad=0.5,
                            handletextpad=0.5,
                            prop={'size': 8, 'family': 'monospace'})  # Use monospace font for consistent spacing
    
    plt.savefig(f'figs/playlists_AV_{color}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def normalize_dfs(df, emo_df):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    df_norm = df.copy()
    df_norm['arousal'] = scaler.fit_transform(df_norm[['Energy']])
    df_norm['valence'] = scaler.fit_transform(df_norm[['Happy']])

    emo_df_norm = emo_df.copy()
    emo_df_norm['valence'] = scaler.fit_transform(emo_df_norm[['valence']])
    emo_df_norm['arousal'] = scaler.fit_transform(emo_df_norm[['arousal']])
    
    spotify_quadrants = count_quadrants(df_norm, 'valence', 'arousal')
    emo_quadrants = count_quadrants(emo_df_norm, 'valence', 'arousal')

    # Convert to DataFrame for display
    freq_df = pd.DataFrame({'Spotify': spotify_quadrants, 'Music2Emo': emo_quadrants})
    print(freq_df)
    print(freq_df / df.shape[0] * 100)

    # get orders from track names
    emo_df_norm[['#', 'Spotify Track Id', 'playlist']] = emo_df_norm['song'].apply(lambda x: pd.Series(extract_order_spotify_id(x)))
    return df_norm, emo_df_norm

# def plot_playlists(df, emo_df, playlists):
#     n = len(playlists)  # Number of playlists
#     fig, axes = plt.subplots(n, 2, figsize=(12, 1.5 * n), sharex=True, sharey=True)

#     for i, play in enumerate(playlists):
#         # Filter and sort data
#         df_playlist = df[df['playlist'] == play]
#         emo_df_playlist = emo_df[emo_df['playlist'] == play].sort_values(by=['#'])

#         # Merge to find common track orders
#         merged = pd.merge(df_playlist[['#', 'valence', 'arousal']], 
#                           emo_df_playlist[['#', 'valence', 'arousal']], 
#                           on='#', suffixes=('_df', '_emo'))

#         # --- Valence subplot (Left Column) ---
#         axes[i, 0].plot(df_playlist['#'], df_playlist['valence'], label='Spotify Valence', color='blue', linestyle='--')
#         axes[i, 0].plot(emo_df_playlist['#'], emo_df_playlist['valence'], label='Mus2Emo Valence', color='blue', linestyle='-')

#         # Compute and plot Valence trendline
#         if len(merged) > 1:
#             val_coef = np.polyfit(merged['#'], merged['valence_emo'], 1)
#             val_trendline = np.poly1d(val_coef)
#             x_vals = np.linspace(df_playlist['#'].min(), df_playlist['#'].max(), 100)
#             axes[i, 0].plot(x_vals, val_trendline(x_vals), color='black', linestyle=':', label='Trendline')

#             # # Compute Pearson correlation for Valence
#             # val_corr = merged[['valence_df', 'valence_emo']].corr().iloc[0, 1]
#             # axes[i, 0].text(0.8, 0.1, f'ρ = {val_corr:.2f}', transform=axes[i, 0].transAxes, 
#             #                 fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

#         axes[i, 0].set_title(f'Valence - {play}')
#         axes[i, 0].grid(True)

#         # --- Arousal subplot (Right Column) ---
#         axes[i, 1].plot(df_playlist['#'], df_playlist['arousal'], label='Spotify Arousal', color='red', linestyle='--')
#         axes[i, 1].plot(emo_df_playlist['#'], emo_df_playlist['arousal'], label='Mus2Emo Arousal', color='red', linestyle='-')

#         # Compute and plot Arousal trendline
#         if len(merged) > 1:
#             aro_coef = np.polyfit(merged['#'], merged['arousal_emo'], 1)
#             aro_trendline = np.poly1d(aro_coef)
#             x_vals = np.linspace(df_playlist['#'].min(), df_playlist['#'].max(), 100)
#             axes[i, 1].plot(x_vals, aro_trendline(x_vals), color='black', linestyle=':', label='Trendline')

#             # # Compute Pearson correlation for Arousal
#             # aro_corr = merged[['arousal_df', 'arousal_emo']].corr().iloc[0, 1]
#             # axes[i, 1].text(0.8, 0.1, f'ρ = {aro_corr:.2f}', transform=axes[i, 1].transAxes, 
#             #                 fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

#         axes[i, 1].set_title(f'Arousal - {play}')
#         axes[i, 1].grid(True)

#     # Labels and legend
#     for ax in axes[:, 0]:  # Left column (Valence)
#         ax.set_ylabel('V')
#     for ax in axes[:, 1]:  # Right column (Arousal)
#         ax.set_ylabel('A')

#     for ax in axes[-1, :]:  # X-labels only for the last row
#         ax.set_xlabel('Track Order (#)')

#     # Adjust layout
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))
#     fig.suptitle('Arousal & Valence Over Time Across Playlists', fontsize=14)
#     fig.tight_layout(rect=[0, 0, 1, 0.98])
#     fig.savefig('figs/time_var.png')

#     # plt.show()

def format_hour(x, _):
    h = int(x)
    m = int(round((x - h) * 60))
    return f"{h}:{m:02d}"


def plot_playlists_time_based(df, emo_df, playlists):
    
    n = len(playlists)
    fig, axes = plt.subplots(n, 2, figsize=(6, 1 * n), sharex=False, sharey=True)

    for i, play in enumerate(playlists):
        df_playlist = df[df['playlist'] == play].copy()
        emo_df_playlist = emo_df[emo_df['playlist'] == play].copy()

        # Sort and compute time from df only
        df_playlist = df_playlist.sort_values(by='#')
        emo_df_playlist = emo_df_playlist.sort_values(by='#')
        emo_df_playlist['#'] += 1
        df_playlist['start_time'] = df_playlist['Duration (s)'].cumsum() - df_playlist['Duration (s)']
        df_playlist['end_time'] = df_playlist['start_time'] + df_playlist['Duration (s)']
        df_playlist['start_time'] /= 3600
        df_playlist['end_time'] /= 3600

        # Merge to align valence/arousal values from emo_df to df
        merged = pd.merge(df_playlist[['#', 'start_time', 'end_time', 'valence', 'arousal', 'phase']],
                          emo_df_playlist[['#', 'valence', 'arousal']],
                          on='#', suffixes=('_df', '_emo'))

        # ----- VALENCE -----
        for _, row in df_playlist.iterrows():
            axes[i, 0].hlines(y=row['valence'], xmin=row['start_time'], xmax=row['end_time'], alpha=0.3,
                              color='blue', linestyle='-', label='Spotify Val.' if row.name == df_playlist.index[0] else "")
        for _, row in merged.iterrows():
            axes[i, 0].hlines(y=row['valence_emo'], xmin=row['start_time'], xmax=row['end_time'],
                              color='blue', linestyle='-', label='Mus2Emo Val.' if row.name == merged.index[0] else "")

        # # Valence trendline
        # if len(merged) > 1:
        #     times = (merged['start_time'] + merged['end_time']) / 2
        #     val_trendline = np.poly1d(np.polyfit(times, merged['valence_emo'], 1))
        #     x_vals = np.linspace(times.min(), times.max(), 100)
        #     axes[i, 0].plot(x_vals, val_trendline(x_vals), 'k:', label='Trendline')

        # axes[i, 0].set_title(f'Valence - {play}')
        axes[i, 0].grid(True)

        # ----- PHASE BACKGROUND -----
        colors = ['r', 'g', 'b']
        for j, phase in enumerate(df_playlist['phase'].unique()):
            phase_df = df_playlist[df_playlist['phase'] == phase]
            start = phase_df['start_time'].min()
            end = phase_df['end_time'].max()
            axes[i, 0].axvspan(start, end, color=colors[j], alpha=0.15, label=phase)


        # ----- AROUSAL -----
        for _, row in df_playlist.iterrows():
            axes[i, 1].hlines(y=row['arousal'], xmin=row['start_time'], xmax=row['end_time'], alpha=0.3,
                              color='red', linestyle='-', label='Spotify Aro.' if row.name == df_playlist.index[0] else "")
        for _, row in merged.iterrows():
            axes[i, 1].hlines(y=row['arousal_emo'], xmin=row['start_time'], xmax=row['end_time'],
                              color='red', linestyle='-', label='Mus2Emo Aro.' if row.name == merged.index[0] else "")

        # # Arousal trendline
        # if len(merged) > 1:
        #     times = (merged['start_time'] + merged['end_time']) / 2
        #     aro_trendline = np.poly1d(np.polyfit(times, merged['arousal_emo'], 1))
        #     x_vals = np.linspace(times.min(), times.max(), 100)
        #     axes[i, 1].plot(x_vals, aro_trendline(x_vals), 'k:', label='Trendline')

        # axes[i, 1].set_title(f'Arousal - {play}')
        axes[i, 1].grid(True)
        axes[i, 1].set_title(f'{play}', fontsize=12, loc='right')

        # ----- PHASE BACKGROUND -----
        colors = ['r', 'g', 'b']
        for j, phase in enumerate(df_playlist['phase'].unique()):
            phase_df = df_playlist[df_playlist['phase'] == phase]
            start = phase_df['start_time'].min()
            end = phase_df['end_time'].max()
            axes[i, 1].axvspan(start, end, color=colors[j], alpha=0.15, label=phase)

        # Inside the loop, after plotting
        axes[i, 0].xaxis.set_major_formatter(mticker.FuncFormatter(format_hour))
        axes[i, 1].xaxis.set_major_formatter(mticker.FuncFormatter(format_hour))


    # Axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel('Valence')
    for ax in axes[:, 1]:
        ax.set_ylabel('Arousal')
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)')

    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[0, 1].get_legend_handles_labels()

    # Merge and deduplicate
    by_label = dict(zip(labels0 + labels1, handles0 + handles1))

    # Adjust figure size and layout to make space for legend on the right
    fig.set_size_inches(9, 1.5 * n)  # Wider canvas
    fig.subplots_adjust(right=0.3)  # Leave space for the legend

    # Add the legend outside the plot area
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='center left',
        bbox_to_anchor=(0.80, 0.5),  # Push slightly inside canvas
        borderaxespad=0.,
        frameon=True,
        fontsize='small'
    )

    # Title and layout
    # fig.suptitle('Valence and Arousal over Time', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.78, 0.95])  # Adjust to avoid overlap with title and legend
    fig.savefig('figs/arousal_valence_time_var.pdf', bbox_inches='tight', dpi=300) 


def plot_playlists_by_feat(df, feats, playlists, feat_name):
    n = len(playlists)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n), sharex=False, sharey=True)

    for i, play in enumerate(playlists):
        df_playlist = df[df['playlist'] == play].copy()
        feats_playlist = feats[feats['playlist'] == play].copy()

        merged_df = df_playlist.merge(feats_playlist, left_on='Spotify Track Id', right_on='spotify_id', how='left')

        # Sort and compute time from df only
        df_playlist = df_playlist.sort_values(by='#')
        df_playlist['start_time'] = df_playlist['Duration (s)'].cumsum() - df_playlist['Duration (s)']
        df_playlist['end_time'] = df_playlist['start_time'] + df_playlist['Duration (s)']
        df_playlist['start_time'] /= 3600
        df_playlist['end_time'] /= 3600

        merged_df['start_time'] = merged_df.groupby('playlist_x').cumcount() * 30 / 3600  # in hours
        merged_df['end_time'] = merged_df['start_time'] + 30 / 3600  # 30s later
         # ----- songs -----
        for _, row in df_playlist.iterrows():
            axes[i].axvline(x=row['start_time'], color='gray', linestyle='--', linewidth=0.5, alpha=0.9)

        for _, row in merged_df.iterrows():
            axes[i].hlines(y=row[feat_name], xmin=row['start_time'], xmax=row['end_time'],
                              color='blue', linestyle='-', label=feat_name if row.name == merged_df.index[0] else "")

        axes[i].set_title(f'{feat_name} - {play}')
        axes[i].grid(True)

        # ----- PHASE BACKGROUND -----
        colors = ['r', 'g', 'b']
        for j, phase in enumerate(merged_df['phase_x'].unique()):
            phase_df = merged_df[merged_df['phase_x'] == phase]
            start = phase_df['start_time'].min()
            end = phase_df['end_time'].max()
            axes[i].axvspan(start, end, color=colors[j], alpha=0.15, label=phase)

        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(format_hour))

    for ax in axes:
        ax.set_ylabel(f'Ac. feat.')

    axes[-1].set_xlabel('Time (h)')

    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles = sum([hl[0] for hl in handles_labels], [])
    labels = sum([hl[1] for hl in handles_labels], [])

    # Deduplicate by label
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='center left',
        bbox_to_anchor=(0.80, 0.5),  
        borderaxespad=0.,
        frameon=True,
        fontsize='small'
    )

    fig.set_size_inches(16, 1.8 * n)  # Wider canvas
    fig.subplots_adjust(right=0.78) 

    # Title and layout
    # fig.suptitle(f'{feat_name} over time across playlists', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.78, 0.95])  #
    fig.savefig(f'figs/feat_{feat_name}_time_var.pdf', bbox_inches='tight', dpi=300) 


if __name__ == '__main__':

    df = pd.read_csv('data/full_data.csv')
    df = df[df['process?'] == True].copy()

    emo_df = get_emotions()
    df, emo_df = normalize_dfs(df, emo_df)

 
    plot_arousal_valence_compact(df, emo_df, 'playlist')
    test = emo_df.merge(df[['#', 'playlist', 'phase']], on=['#', 'playlist'], how='left')
    plot_arousal_valence_compact(df, test, 'phase')


    playlists = df['playlist'].unique()

    plot_playlists_time_based(df, emo_df, playlists)

    feats = pd.read_csv('data/df_compare_lld.csv', index_col=0)

    plot_playlists_by_feat(df, feats, playlists, 'BPM')
    plot_playlists_by_feat(df, feats, playlists, 'pcm_RMSenergy_sma')
    plot_playlists_by_feat(df, feats, playlists, 'F0final_sma')
    plot_playlists_by_feat(df, feats, playlists, 'audspec_lengthL1norm_sma')
    plot_playlists_by_feat(df, feats, playlists, 'mfcc_sma[11]')
    plot_playlists_by_feat(df, feats, playlists, 'mfcc_sma[3]')
    plot_playlists_by_feat(df, feats, playlists, 'mfcc_sma[5]')
   






