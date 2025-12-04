import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_rel

import pdb

def classify(df, clf, cv):
    seed = np.random.seed(1987)
    feature_columns = [col for col in df.columns if col not in 
                        ['#', 'Song', 'Artist', 'Time', 'Genres', 'Parent Genres', 'Album',
                        'Album Date', 'Added At', 'Spotify Track Id', 'Album Label', 'Camelot',
                        'ISRC', 'Duration (s)', 'Duration', 'phase', 'playlist', 'Time Duration',
                        'Start Time', 'Phase']]
    print(f'Using {feature_columns} for classification!')

    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f'Numeric features {numeric_cols} for classification!')
    print(f'Categorical features {categorical_cols} for classification!')

    X = df[feature_columns]
    y = df[clf].values 

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(zip(np.unique(y), class_weights))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', class_weight=class_weight_dict))
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
    y_true_all, y_pred_all, y_score_all = [], [], []
    model_accs_per_fold = []
    baseline_accs_per_fold = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, y_pred)
        model_accs_per_fold.append(model_acc)
        y_score = model.predict_proba(X_test)

        majority_class = Counter(y_train).most_common(1)[0][0]
        y_baseline_pred = np.full_like(y_test, majority_class)
        baseline_acc = accuracy_score(y_test, y_baseline_pred)
        baseline_accs_per_fold.append(baseline_acc)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_score_all.extend(y_score)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_score_all = np.array(y_score_all)

    plot_roc_auc(y_true_all, y_score_all)
    compare_with_chance(model, y_true_all, y_pred_all, model_accs_per_fold, baseline_accs_per_fold)


def plot_roc_auc(y_true, y_score):
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_bin.shape[1]
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    

def compare_with_chance(model, y_true, y_pred, model_accs_per_fold=None, baseline_accs_per_fold=None):
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm = np.round(cm, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap='Blues',  
        include_values=True,     
        colorbar=False,
    )

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    for i in range(len(model.classes_)):
        for j in range(len(model.classes_)):
            text = disp.text_[i, j]
            text.set_fontsize(6)
            if i == j:
                text.set_weight('bold')  

    ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.show()
    majority_class = Counter(y_true).most_common(1)[0][0]
    y_chance_pred = np.full_like(y_true, majority_class)

    model_acc = accuracy_score(y_true, y_pred)
    chance_acc = accuracy_score(y_true, y_chance_pred)

    print(f"Model Accuracy: {model_acc:.3f}")
    print(f"Chance Baseline Accuracy: {chance_acc:.3f}\n")

    print("Model Classification Report:")
    print(classification_report(y_true, y_pred))

    # print("Baseline (Chance) Classification Report:")
    # print(classification_report(y_true, y_chance_pred))

    # Perform paired t-test with cross-validation
    if model_accs_per_fold is not None and baseline_accs_per_fold is not None:
        # Convert to numpy arrays for easier computation
        model_accs = np.array(model_accs_per_fold)
        baseline_accs = np.array(baseline_accs_per_fold)
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(model_accs, baseline_accs)
        
        print(f"\nPaired t-test (cross-validation):")
        print(f"  Model mean accuracy across folds: {model_accs.mean():.3f} ± {model_accs.std():.3f}")
        print(f"  Baseline mean accuracy across folds: {baseline_accs.mean():.3f} ± {baseline_accs.std():.3f}")
        print(f"  t-statistic: {t_stat:.3f}")
        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""
        
        print(f"  p-value: {p_value:.7f}{sig_marker}")

        if p_value < 0.05 and model_accs.mean() > baseline_accs.mean():
            print("The classifier is significantly better than chance! 🎉")
        elif p_value < 0.05 and model_accs.mean() < baseline_accs.mean():
            print("The classifier is significantly worse than chance. 😡")
        else:
            print("The classifier is NOT significantly better than chance. 🤔")
    else:
        print(f"\nNote: Per-fold accuracies not provided. Cannot perform paired t-test.")
        if model_acc > chance_acc:
            print("The classifier has higher accuracy than chance baseline.")
        elif model_acc < chance_acc:
            print("The classifier has lower accuracy than chance baseline.")
        else:
            print("The classifier has the same accuracy as chance baseline.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify psilocybin lists")
    parser.add_argument('-clf', 
                        type=str,
                        choices=['playlist', 'phase'],
                        default='playlist',
                        help='Select labels to choose for classification.')
    parser.add_argument('-cv', 
                        type=int,
                        choices=[5, 10],
                        default=5,
                        help='Select a number of cross-validation splits')
    args = parser.parse_args()
    playlists = ['chacruna_baldwin', 'chacruna_kelan_thomas2', 'compass_v2',
                 'copenhagen', 'imperial1', 'imperial2', 'jh_classical', 'jh_overtone']
    base_path = '/Users/jsgomezc/Data/psilocybin'
    df = pd.read_csv('data/full_data.csv')
    df = df[df['process?'] == True].copy()

    classify(df, args.clf, args.cv)