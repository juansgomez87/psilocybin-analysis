# Process Psilocybin playlists

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






