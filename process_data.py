import soundfile as sf
import glob
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
from scipy.signal import resample_poly
import gc
import multiprocessing as mp
from functools import partial
# import torch
# import objgraph

import opensmile
# from maest import get_maest
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import psutil



import pdb

# from Music2Emotion.music2emo import Music2emo
# music2emo = Music2emo()


def process_compare_lld(fn, out_fn, smile_obj):
    """Process audio file and extract features using opensmile (LLD version with averaging)"""
    
    dir_path = os.path.dirname(out_fn)
    os.makedirs(dir_path, exist_ok=True)

    data, samplerate = sf.read(fn)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data_resampled = resample_poly(data, up=16000, down=samplerate)
    chunk_size = 30 * 16000  # 30 sec * 16 kHz
    stride = chunk_size

    # Process chunks
    embeddings = []
    for start in range(0, len(data_resampled), stride):
        
        chunk = data_resampled[start:start + chunk_size]

        if len(chunk) < chunk_size:  # Zero-pad the last chunk if needed
            # chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            continue # avoid zero padding and skip last chunk

        this_emb = smile_obj.process_signal(chunk, 16000).values
        embeddings.append(this_emb.mean(axis=0))

        gc.collect()
        # print(f"Memory usage after: {psutil.Process().memory_info().rss / (1024 ** 2)} MB")

    # Combine embeddings
    embeddings = np.vstack(embeddings)
    this_feat = pd.DataFrame(
        embeddings, 
        columns=smile_obj.feature_names,
        index=pd.MultiIndex.from_product([[fn], range(embeddings.shape[0])], names=["file", "chunk"])
    )
    this_feat.to_csv(out_fn)


def process_single_file(args):
    """Helper function to process a single file for multiprocessing"""
    f, path_audio, path_compare_lld_feats = args
    out_f = f.replace(path_audio, path_compare_lld_feats).replace('.mp3', '.csv')
    
    if os.path.exists(out_f):
        return f"Already exists: {f}"
    
    try:
        # Create smile object in worker process
        worker_smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        
        process_compare_lld(f, out_f, worker_smile)
 
        return f"Success: {f}"
    except Exception as e:
        return f"Error processing {f}: {str(e)}"


def process_playlist_parallel(playlist, df, base_path, n_processes=None):
    """Process a playlist using multiprocessing"""
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    path_audio = os.path.join(base_path, 'audio', playlist)
    path_compare_lld_feats = os.path.join(base_path, 'compare_lld', playlist)
    all_mp3 = glob.glob(os.path.join(path_audio, '*.mp3'))

    codes_to_process = df[(df['playlist'] == playlist) & (df['process?'] == True)]['Spotify Track Id'].tolist()
    mp3_to_process = [f for f in all_mp3 if f.split('/')[-1].split('-')[1] in codes_to_process]

    
    print(f'Processing {len(mp3_to_process)} files from {len(all_mp3)} using {n_processes} processes...')
    
    # Prepare arguments for multiprocessing
    args_list = [(f, path_audio, path_compare_lld_feats) for f in mp3_to_process]
    
    # Use multiprocessing to process files in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc=f'Processing {playlist}',
            colour="red"
        ))
    
    return results


if __name__ == '__main__':
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process audio files with multiprocessing')
    parser.add_argument('--n-processes', type=int, default=None, 
                       help='Number of processes to use (default: CPU count)')
    args = parser.parse_args()
    
    base_path = '/Users/jsgomezc/Data/psilocybin'
    df = pd.read_csv('data/full_data.csv')
    playlists = df.playlist.unique()
    print(f"Using {args.n_processes or mp.cpu_count()} processes")
    
    for playlist in tqdm(playlists, f'Processing playlists...', colour="red"):
        process_playlist_parallel(playlist, df, base_path, args.n_processes)
