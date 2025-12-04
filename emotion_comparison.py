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
import colorsys

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


def plot_arousal_valence_compact(df, emo_df, color='playlist'):

    all_categories = df[color].unique()
    playlists = all_categories
    n_colors = len(playlists)
    colors = [colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.8) for i in range(n_colors)]
    colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                for r, g, b in colors]
    colors = dict(zip(playlists, colors))
    

    fig = plt.figure(figsize=(8, 3))  
    
    ax1 = fig.add_axes([0.07, 0.12, 0.34, 0.8])  
    ax2 = fig.add_axes([0.46, 0.12, 0.34, 0.8])  
    ax2.sharey(ax1)  
    legend_ax = fig.add_axes([0.82, 0.12, 0.16, 0.8])  
    legend_ax.axis('off')
    
    ax1.scatter(df['valence'], df['arousal'], c=df[color].map(colors), s=20, alpha=0.4)
    ax1.set(xlabel='Valence', ylabel='Arousal', xlim=(-1, 1), ylim=(-1, 1), title='Spotify Data')
    ax1.set_xlabel('Valence', fontsize=12)
    ax1.set_ylabel('Arousal', fontsize=12)
    ax1.set_title('Spotify', fontsize=14)
    ax1.tick_params(labelsize=11)
    ax1.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(x=0.0, color='black', linestyle='--', linewidth=1)
    
    emo_df_clean = emo_df.dropna()
    ax2.scatter(emo_df_clean['valence'], emo_df_clean['arousal'], 
               c=emo_df_clean[color].map(colors), s=20, alpha=0.4)
    ax2.set(xlabel='Valence', xlim=(-1, 1), ylim=(-1, 1), title='Mus2Emo Data')
    ax2.set_xlabel('Valence', fontsize=12)
    ax2.set_title('Mus2Emo', fontsize=14)
    ax2.tick_params(labelsize=11)
    ax2.set_ylabel('')  
    plt.setp(ax2.get_yticklabels(), visible=False)  
    ax2.axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=0.0, color='black', linestyle='--', linewidth=1)
    
    max_length = 22  
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, 
                         markersize=10, label=f"{p.replace('_', ' '):<{max_length}}") for p, c in colors.items()]
    
    while len(handles) < 8:
        dummy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w', 
                          markersize=0, label=' ' * max_length)
        handles.append(dummy)
    
    legend = legend_ax.legend(handles=handles, 
                            title=color.replace('_', ' ').title(),
                            loc='center left',
                            borderpad=0.5,
                            handletextpad=0.5,
                            prop={'size': 11, 'family': 'Courier New'})  
    legend.get_title().set_fontsize(13)  
    
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

    freq_df = pd.DataFrame({'Spotify': spotify_quadrants, 'Music2Emo': emo_quadrants})
    print(freq_df)
    print(freq_df / df.shape[0] * 100)

    emo_df_norm[['#', 'Spotify Track Id', 'playlist']] = emo_df_norm['song'].apply(lambda x: pd.Series(extract_order_spotify_id(x)))
    return df_norm, emo_df_norm

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

        df_playlist = df_playlist.sort_values(by='#')
        emo_df_playlist = emo_df_playlist.sort_values(by='#')
        emo_df_playlist['#'] += 1
        df_playlist['start_time'] = df_playlist['Duration (s)'].cumsum() - df_playlist['Duration (s)']
        df_playlist['end_time'] = df_playlist['start_time'] + df_playlist['Duration (s)']
        df_playlist['start_time'] /= 3600
        df_playlist['end_time'] /= 3600

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

        axes[i, 0].grid(True)

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

        axes[i, 1].grid(True)
        axes[i, 1].set_title(f'{play}', fontsize=12, loc='right')

        # ----- PHASE BACKGROUND -----
        colors = ['r', 'g', 'b']
        for j, phase in enumerate(df_playlist['phase'].unique()):
            phase_df = df_playlist[df_playlist['phase'] == phase]
            start = phase_df['start_time'].min()
            end = phase_df['end_time'].max()
            axes[i, 1].axvspan(start, end, color=colors[j], alpha=0.15, label=phase)

        axes[i, 0].xaxis.set_major_formatter(mticker.FuncFormatter(format_hour))
        axes[i, 1].xaxis.set_major_formatter(mticker.FuncFormatter(format_hour))


    for ax in axes[:, 0]:
        ax.set_ylabel('Valence')
    for ax in axes[:, 1]:
        ax.set_ylabel('Arousal')
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)')

    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[0, 1].get_legend_handles_labels()

    by_label = dict(zip(labels0 + labels1, handles0 + handles1))

    fig.set_size_inches(9, 1.5 * n)  
    fig.subplots_adjust(right=0.3)  

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='center left',
        bbox_to_anchor=(0.80, 0.5),  
        borderaxespad=0.,
        frameon=True,
        fontsize='small'
    )

    fig.tight_layout(rect=[0, 0, 0.78, 0.95])  
    fig.savefig('figs/arousal_valence_time_var.pdf', bbox_inches='tight', dpi=300) 


def plot_playlists_by_feat(df, feats, playlists, feat_name):
    n = len(playlists)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n), sharex=False, sharey=True)

    for i, play in enumerate(playlists):
        df_playlist = df[df['playlist'] == play].copy()
        feats_playlist = feats[feats['playlist'] == play].copy()

        merged_df = df_playlist.merge(feats_playlist, left_on='Spotify Track Id', right_on='spotify_id', how='left')

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

    fig.set_size_inches(16, 1.8 * n)  
    fig.subplots_adjust(right=0.78) 

    fig.tight_layout(rect=[0, 0, 0.78, 0.95])  
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
   






