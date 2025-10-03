# Acoustic characterization of music for psilocybin therapy

Music plays a central role in psilocybin therapy, guiding participants through onset, peak, and return phases of the psychedelic experience. Playlists are typically designed to be supportive and anxiolytic over the 4–6 hour course of a medium-to-high dose, yet no established guidelines exist for content selection, and it is unclear whether playlists share systematic musical, acoustic, or emotional features. We applied computational music analysis to eight publicly available psychedelic therapy playlists, examining high-level musical features (Spotify API), low-level acoustic descriptors (ComParE), and modeled arousal and valence (Music2Emo). Classification models (logistic regression, random forests) tested differences across playlists and phases using both song-level and 30-second segment-level features. Results showed no consistent patterns differentiating onset, peak, and return across playlists, suggesting sequencing is not guided by shared principles. Within playlists, however, energy- and timbre-related features distinguished phases, indicating reliance on individual heuristics rather than common constructs. Segment-level features improved phase classification within playlists, while song-level features improved classification between playlists. Overall, this first large-scale acoustic study of psilocybin therapy music suggests limited consistency in current curation practices and highlights the need for more intentional structuring of music to support therapeutic goals.

### Installation 
Install the required dependencies for Python 3.11.13:
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### To run Music2Emo locally
Install the required dependencies for Python 3.11.13:
```
python3.11 -m venv .musenv
source .musenv/bin/activate
git clone git@github.com:AMAAI-Lab/Music2Emotion.git
cd Music2Emotion
pip install -r requirements.txt
cd ..
python music2emo.py
```

### Usage for Spotify Features
1. To make the plots comparing music2emo and spotify:
```
python emotion_comparison.py
```

2. Run classification using Spotify features:
```
python spoti_clf.py --clf [playlist/phase]
```

### Usage for acoustic features Co

1. Extract all acoustic features from the data:
```
python process_data.py --n-process 10
```

2. Assemble all features for classification.
```
python assemble_data.py
```

3. Run classifier script to obtain classifiers (log - logistic regression, rf - random forest) and evaluation. 
```
python classifier.py -clf [phase/playlist] -algo compare_lld -mean [y/n] -reg [log/rf]
```

4. Run stats script to obtain statistics on features. We do this only on features every 30s.
```
python stats.py -algo compare_lld
```






