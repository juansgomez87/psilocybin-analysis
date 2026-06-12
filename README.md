# Music for Psilocybin Therapy and the Temporal Arc of Psychedelic Experience
Juan Sebastián Gómez-Cañón, Carly Leininger, Danielle Mayall, Robert F. Dougherty, Daniel L. Bowling

### Significance statement
In this first large-scale acoustic analysis of music for psilocybin therapy (PT), we show that commonly used playlists do not align with the pharmacodynamic onset–peak–return model of psychedelic experience.
These findings indicate an absence of acoustically coherent therapeutic goals, motivating the development of new approaches to therapeutic music design that aim to combine structured acoustic features with context sensitive adaptation.

### Abstract
Music plays a central role in psilocybin therapy (PT), where playlists are designed to be supportive and stimulating over a 4–6-hour psychedelic experience.
The primary design principle underlying PT music programming is that music should be affectively aligned with the onset–peak–return model of pharmacodynamic action.
However, in the absence of clear acoustic criteria, this framework has been implemented largely through individual or team-based curatorial judgment, making it unclear whether contemporary research PT playlists consistently realize the proposed onset–peak–return progression.
Here, we test this question by analyzing 46.9 hours of audio from eight PT music playlists (369 tracks) for systematic acoustic and affective organization across onset, peak, and return phases.
To this end, we applied supervised and unsupervised machine learning approaches to a comprehensive set of audio features extracted from PT music playlist tracks, evaluating phase differentiation across the full dataset as well as within individual playlists.
The results demonstrate that although weak phase-based differentiation is apparent for low-level audio features, especially within some playlists, the overall set lacks consistent phase-based structure.
This implies that current curational practices are not aligned with their theoretical basis, raising questions about how such practices can evolve to better support emerging psychedelic treatments.

## Repository structure

| Script | Purpose |
| --- | --- |
| `process_data.py` | Extract low-level acoustic features (openSMILE) from playlist audio. |
| `assemble_data.py` | Assemble extracted features into per-chunk and per-song datasets. |
| `classifier.py` | Supervised classification of phase / playlist (logistic regression, random forest). |
| `cluster.py` | Unsupervised clustering of phase / playlist (k-means, GMM, agglomerative). |
| `stats.py` | One-way ANOVA on features across phases (computed on 30 s segments). |
| `music2emo.py` | Predict per-track valence/arousal with the Music2Emotion model. |
| `emotion_comparison.py` | Compare Music2Emo and Spotify affective features; generate comparison plots. |
| `spoti_clf.py` | Classify phase / playlist from Spotify audio features. |

Intermediate and final datasets live in `data/` (see [Data](#data)).

## Installation
Requires Python 3.11.13. Create the main environment and install dependencies:
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Music2Emo runs in a **separate** environment because it has its own dependencies:
```
python3.11 -m venv .musenv
source .musenv/bin/activate
git clone git@github.com:AMAAI-Lab/Music2Emotion.git
cd Music2Emotion
pip install -r requirements.txt
cd ..
```

## Data
The `data/` directory holds the feature tables used by the analysis scripts:

| File | Description |
| --- | --- |
| `full_data.csv` | Track-level metadata (playlist, phase, duration, links). |
| `df_compare_lld.csv` | Per-chunk acoustic features (low-level descriptors). |
| `df_compare_lld_mean.csv` | Per-song (mean) acoustic features. |
| `music_2_emo.csv` | Music2Emo per-track valence/arousal predictions. |
| `30s_music_2_emo.csv` | Music2Emo predictions on 30 s segments. |

The raw audio files are **not** included. `process_data.py` and `music2emo.py`
require local audio; the analysis, classification, clustering, and statistics
steps run directly on the provided CSVs.

## Usage

### Acoustic feature pipeline
1. Extract acoustic features from the raw audio (requires local audio):
   ```
   python process_data.py --n-processes 10
   ```
2. Assemble the extracted features into the classification datasets:
   ```
   python assemble_data.py
   ```
3. Supervised classification. `-clf` selects the labels, `-reg` the model
   (`log` = logistic regression, `rf` = random forest). Add `-mean` for
   song-level features and `-plot` to save figures:
   ```
   python classifier.py -clf [phase/playlist] -reg [log/rf] [-mean] [-plot]
   ```
4. Unsupervised clustering. `-label` selects the labels to compare against and
   `-method` the algorithm:
   ```
   python cluster.py -label [phase/playlist] -method [kmeans/gmm/agglomerative] [-mean] [-plot]
   ```
5. Statistics (one-way ANOVA on features, computed on 30 s segments):
   ```
   python stats.py -clf [playlist/phase]
   ```

### Affective features (Spotify and Music2Emo)
1. Predict per-track emotion with Music2Emo (run in the `.musenv` environment):
   ```
   python music2emo.py
   ```
2. Compare Music2Emo and Spotify affective features and generate the plots:
   ```
   python emotion_comparison.py
   ```
3. Classify phase / playlist from Spotify audio features:
   ```
   python spoti_clf.py -clf [playlist/phase]
   ```

## Citation
If you use this code, please cite the accompanying paper (citation to be added on
publication).
