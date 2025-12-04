# Do Psilocybin Therapy Playlists align with Psychedelic Experience? An empirical test of acoustic and affective trajectories
Juan Sebastián Gómez-Cañón, Carly Leininger, Danielle Mayall, Robert F. Dougherty, Daniel L. Bowling

Music plays a central role in psilocybin therapy, guiding participants through onset, peak, and return phases of the psychedelic experience. 
Playlists are theoretically designed to be supportive and anxiolytic over the 4–6 hour course of a medium-to-high dose but, yet no established guidelines exist for content selection, and it is unclear whether playlists in fact share systematic musical, acoustic, or emotional features remains unclear.
We applied computational music analysis to eight publicly available psychedelic therapy playlists, examining high-level musical features via the Spotify application programming interface, low-level acoustic features from the ComParE feature set, and estimates of perceived arousal and valence calculated using the Music2Emo model. 
Supervised and unsupervised machine learning algorithms were applied to test for consistency of differences across onset, peak, and return playlist phases at both the track-level and 30-second segment-level. 
The results indicate an absence of consistent patterning differentiating onset, peak, and return phases across playlists, suggesting that sequencing is not guided by shared musical, acoustic, or emotional principles. 
Within playlists, however, energy- and timbre-related features did distinguish phases, suggesting that curators may rely on internally coherent heuristics rather than theoretical, cross-playlist constructs.
This first large-scale acoustic study of common music playlists for psilocybin therapy suggests limited consistency in current curation practices as well as limited alignment with established theories of music’s role in psychedelic treatment, raising questions about optimization in support of clinical goals. 

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

3. Run unsupervised learning script to obtain clusters and evaluation. 
```
python cluster.py -label [phase/playlist] -algo compare_lld -mean [y/n] -method [kmeans/gmm/agglomerative]
```

4. Run stats script to obtain statistics on features. We do this only on features every 30s.
```
python stats.py -algo compare_lld
```






