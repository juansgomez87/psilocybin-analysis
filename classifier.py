"""Supervised classification of phase or playlist from acoustic features (logistic regression / random forest).

Run: python classifier.py -clf [phase/playlist] -reg [log/rf] [-mean] [-plot]
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
from scipy.stats import ttest_rel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import random

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in matmul') 
warnings.filterwarnings('ignore', message='divide by zero encountered in matmul')
warnings.filterwarnings('ignore', message='overflow encountered in matmul') 
warnings.filterwarnings('ignore', message='invalid value encountered in divide')


class PsiloClassifier():
    def __init__(self, clf, algo, mean, cv, plot_flag, playlist, reg, resample_method=None, clip=False, boundaries=False, do_pca=False, return_metrics=False, save_svg=False):
        self.seed = 1987
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.do_pca = do_pca
        self.clf = clf
        self.algo = algo
        self.df = pd.read_csv(f'data/df_{algo}{mean}.csv', index_col=0)
        self.plot_flag = plot_flag
        self.playlist = playlist
        self.reg = reg  # log - logistic regression, rf - random forest
        self.resample_method = resample_method
        self.clip = clip
        self.boundaries = boundaries
        self.return_metrics = return_metrics
        self.save_svg = save_svg
        self.results = None
        self._fig_dir = "svgs" if save_svg else "figs"
        self._fig_ext = ".svg" if save_svg else ".pdf"
        if save_svg:
            os.makedirs(self._fig_dir, exist_ok=True)

        if self.clf == 'phase' and self.boundaries:
            # always find songs at the boundaries of each phase for whole songs
            temp = pd.read_csv(f'data/df_{algo}_mean.csv', index_col=0)
            temp = temp.reset_index(drop=True)
            # from onset to peak
            boundaries_onset_peak = [(i, i+1) for i in range(len(temp) - 1) if temp.loc[i, 'phase'] == 'onset' and temp.loc[i+1, 'phase'] == 'peak']
            # flatten the list of tuples
            boundaries_onset_peak = [i for t in boundaries_onset_peak for i in t]
            # from peak to return
            boundaries_peak_return = [(i, i+1) for i in range(len(temp) - 1) if temp.loc[i, 'phase'] == 'peak' and temp.loc[i+1, 'phase'] == 'return']
            # flatten the list of tuples
            boundaries_peak_return = [i for t in boundaries_peak_return for i in t]
            # get the index of the songs at the boundaries
            boundaries_idx = boundaries_onset_peak + boundaries_peak_return
            # get spotify ids 
            spotify_ids = temp.loc[boundaries_idx, 'spotify_id'].values
            # drop all rows with that spotify id
            initial_len = len(self.df)
            self.df = self.df[~self.df['spotify_id'].isin(spotify_ids)]
            removed_len = initial_len - len(self.df)
            if mean:
                print(f'Removing {removed_len} songs at the boundaries of each phase for whole songs!')
            else:
                print(f'Removing {removed_len} chunks at the boundaries of each phase for chunks!')


        if self.clf == 'phase' and self.clip:
            print('Clipping the data of return phase!')
            # use full data to get durations and clip the return phase
            temp = pd.read_csv(f'data/full_data.csv', index_col=0)
            temp = temp[temp['process?'] == True].copy()

            return_durations = [(_, temp[(temp.playlist == _) & (temp.phase == 'return')]['Duration (m)'].sum()) for _ in temp['playlist'].unique()]
            shortest = min(return_durations, key=lambda x: x[1])
            # ('imperial2', np.float64(126.983))
            print(f'Shortest return phase is {shortest[0]} with duration {shortest[1]} minutes!')
            # clip the return phase of all other playlists to the duration of the shortest return phase
            spotify_ids = []
            for playlist in temp['playlist'].unique():
                if playlist != shortest[0]:
                    df_playlist = temp[temp['playlist'] == playlist]
                    df_playlist = df_playlist[df_playlist['phase'] == 'return']
                    df_playlist = df_playlist.reset_index(drop=True)                    
                    # Calculate cumulative sum of durations
                    df_playlist['cumulative_duration'] = df_playlist['Duration (m)'].cumsum()
                    
                    # Find rows that need to be dropped (those that exceed the threshold)
                    rows_to_drop = df_playlist[df_playlist['cumulative_duration'] > shortest[1]]
                    ids = rows_to_drop['link'].str.extract(r'track/([a-zA-Z0-9]+)')[0].tolist()
                    spotify_ids.extend(ids)
                    # print(f'Playlist {playlist}: Dropping {len(ids)} songs with Spotify IDs: {ids}')

            initial_len = len(self.df)
            self.df = self.df[~self.df['spotify_id'].isin(spotify_ids)]
            removed_len = initial_len - len(self.df)
            if mean:
                print(f'Removing {removed_len} songs by clipping return phase!')
            else:
                print(f'Removing {removed_len} chunks by clipping return phase!')
                    
        if playlist == 'all':
            print('Using all playlists!')
            all_playlists = self.df['playlist'].unique().tolist()
            print(all_playlists)
            self.df = self.df[self.df['playlist'].isin(all_playlists)]
        elif playlist == 'most':
            # use all playlists except imperial1
            print('Using all playlists except imperial1!')
            all_playlists = [_ for _ in self.df['playlist'].unique().tolist() if _ != 'imperial1']
            print(all_playlists)
            self.df = self.df[self.df['playlist'].isin(all_playlists)]
        else:
            self.df = self.df[self.df['playlist'] == playlist]
            print(f'Using only {playlist} playlist!')

        # Define feature and label columns
        self.feature_columns = [col for col in self.df.columns if col not in 
                           ['file', 'chunk', 'phase', 'playlist', 'umap_x', 'umap_y', 'artist', 'song', 'spotify_id']]
        

        self.pca_reduction()

        self.y = self.df[clf].values 
        self.cv = cv

        # Initialize model without class weights (will be set per fold)
        if self.reg == 'log':
            self.model = LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                random_state=self.seed
            )
        elif self.reg == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,  
                random_state=self.seed,
                n_jobs=-1
            )
        
        self.run_classification()
        
    def pca_reduction(self):
        
        self.X = self.df[self.feature_columns].values
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
        if self.do_pca:
            print('Calculating PCA, this might take time...')
            pca = PCA(n_components=0.99)
            X_pca = pca.fit_transform(self.X)
            n_comp = X_pca.shape[1]
            print('-'*50)
            print(f'PCA reduced from {len(self.feature_columns)} to {n_comp} to keep 99% of variance!')

            self.feature_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
            df_pca = pd.DataFrame(X_pca, columns=self.feature_columns, index=self.df.index)
            self.df = pd.concat([self.df, df_pca], axis=1)
            self.X = self.df[self.feature_columns].values

    def _resample_fold(self, X_train, y_train):
        """Apply resampling to a single fold's training data and calculate class weights."""
        if self.resample_method == 'smote':
            smote = SMOTE(random_state=self.seed, k_neighbors=min(5, min(Counter(y_train).values())-1))
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        elif self.resample_method == 'under':
            undersampler = RandomUnderSampler(random_state=self.seed, sampling_strategy='auto')
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
        else:
            X_resampled, y_resampled = X_train, y_train
        
        # Calculate class weights for the resampled training data
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_resampled),
            y=y_resampled
        )
        class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))
        
        return X_resampled, y_resampled, class_weight_dict

    def run_classification(self):
        skf = StratifiedGroupKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        
        y_true_all, y_pred_all, y_score_all = [], [], []
        
        model_accs_per_fold = []
        baseline_accs_per_fold = []
        macro_f1_per_fold = []
        for train_idx, test_idx in skf.split(self.X, self.y, groups=self.df['file'].values):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Apply resampling to training data only if specified
            if self.resample_method:
                if not self.return_metrics:
                    print(f"Original training class distribution: {Counter(y_train)}")
                X_train, y_train, class_weight_dict = self._resample_fold(X_train, y_train)
                if not self.return_metrics:
                    print(f"Resampled training class distribution: {Counter(y_train)}")
                    print(f"Test class distribution: {Counter(y_test)}")
                # Update model with fold-specific class weights
                if self.reg == 'log':
                    self.model.set_params(class_weight=class_weight_dict)
                elif self.reg == 'rf':
                    self.model.set_params(class_weight=class_weight_dict)
            else:
                # Calculate class weights for original training data
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )
                class_weight_dict = dict(zip(np.unique(y_train), class_weights))
                if self.reg == 'log':
                    self.model.set_params(class_weight=class_weight_dict)
                elif self.reg == 'rf':
                    self.model.set_params(class_weight=class_weight_dict)
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            model_acc = accuracy_score(y_test, y_pred)
            model_accs_per_fold.append(model_acc)
            macro_f1_per_fold.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            y_score = self.model.predict_proba(X_test)
            
            # Calculate baseline accuracy per fold (majority class from training data)
            majority_class = Counter(y_train).most_common(1)[0][0]
            y_baseline_pred = np.full_like(y_test, majority_class)
            baseline_acc = accuracy_score(y_test, y_baseline_pred)
            baseline_accs_per_fold.append(baseline_acc)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_score_all.extend(y_score)
        
        # Convert lists to numpy arrays
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_score_all = np.array(y_score_all)
        
        if self.return_metrics:
            model_accs = np.array(model_accs_per_fold)
            baseline_accs = np.array(baseline_accs_per_fold)
            macro_f1 = np.array(macro_f1_per_fold)
            t_stat, p_value = ttest_rel(model_accs, baseline_accs)
            cm = confusion_matrix(y_true_all, y_pred_all, labels=self.model.classes_)
            cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            cm_norm = np.round(cm_norm, 2)
            self.results = {
                'accuracy_mean': float(model_accs.mean()),
                'accuracy_std': float(model_accs.std()),
                'baseline_accuracy_mean': float(baseline_accs.mean()),
                'baseline_accuracy_std': float(baseline_accs.std()),
                'macro_f1_mean': float(macro_f1.mean()),
                'macro_f1_std': float(macro_f1.std()),
                't': float(t_stat),
                'p': float(p_value),
                'confusion_matrix': cm_norm,
                'classes': list(self.model.classes_),
            }
            return

        self.plot_stratified_splits()
        self.plot_roc_auc(y_true_all, y_score_all, self.model.classes_)
        self.compare_with_chance(y_true_all, y_pred_all, model_accs_per_fold, baseline_accs_per_fold)
        self.feature_importance()
        if self.plot_flag:
            self.explain_with_shap()
        
    def evaluate(self, y_true, y_pred, y_score):
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
    def plot_stratified_splits(self):
        labels, counts = np.unique(self.y, return_counts=True)
        if self.plot_flag:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=labels, y=counts, color='blue', alpha=0.7)
            plt.xlabel('Classes')
            plt.xticks(labels, rotation=90, fontsize=9)
            plt.ylabel('Count')
            title = 'Class Distribution Before Splitting'
            if self.resample_method:
                title += f' (Resampling: {self.resample_method})'
            plt.title(title)
            plt.tight_layout()
            plt.savefig(f'{self._fig_dir}/{self.clf}_{self.algo}_{self.reg}_class_dist{self._fig_ext}')
            plt.show()
        

        
    def plot_roc_auc(self, y_true, y_score, class_names=None, 
                      axis_labelsize=18, tick_labelsize=14, legend_fontsize=14, title_fontsize=18, curve_labelsize=13):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_bin.shape[1]

        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]

        plt.figure(figsize=(8, 6))
        auc_info = []

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_info.append((roc_auc, fpr, tpr, class_names[i]))

        # Sort by descending AUC
        auc_info.sort(reverse=True, key=lambda x: x[0])

        for roc_auc, fpr, tpr, name in auc_info:
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=axis_labelsize)
        plt.ylabel('True Positive Rate', fontsize=axis_labelsize)
        plt.title('ROC Curve', fontsize=title_fontsize)
        plt.legend(title="Classes", loc='lower right', fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        plt.xticks(fontsize=tick_labelsize)
        plt.yticks(fontsize=tick_labelsize)
        # Set curve label font size
        ax = plt.gca()
        for text in ax.get_legend().get_texts():
            text.set_fontsize(curve_labelsize)
        plt.savefig(f'{self._fig_dir}/{self.clf}_{self.algo}_{self.reg}_roc{self._fig_ext}')
        plt.show()


    def compare_with_chance(self, y_true, y_pred, model_accs_per_fold=None, baseline_accs_per_fold=None):
        cm = confusion_matrix(y_true, y_pred, labels=self.model.classes_)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 2)

        print('-'*50)
        print('Confusion Matrix:')
        print(f'{self.model.classes_}')
        print(cm)
        print('-'*50)

        if self.plot_flag:
            fig, ax = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
            if self.clf != 'playlist':
                disp.plot(
                    ax=ax,
                    xticks_rotation=90,
                    cmap='Blues',  
                    include_values=True,     
                    colorbar=False,
                )
            else:
                disp.plot(
                    ax=ax,
                    xticks_rotation=90,
                    cmap='Blues',  
                    include_values=False,     
                    colorbar=True,
                )  

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            if self.clf != 'playlist':
                for i in range(len(self.model.classes_)):
                    for j in range(len(self.model.classes_)):
                        text = disp.text_[i, j]
                        text.set_fontsize(14)
                        if i == j:
                            text.set_weight('bold')  # bold diagonal
                            # text.set_color('black') 


            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel('Predicted label', fontsize=15)
            ax.set_ylabel('True label', fontsize=15)
            plt.tight_layout()
            plt.savefig(f'{self._fig_dir}/{self.clf}_{self.algo}_{self.reg}_confmat{self._fig_ext}')
            plt.show()

        # Majority class baseline
        majority_class = Counter(y_true).most_common(1)[0][0]
        y_chance_pred = np.full_like(y_true, majority_class)

        # Compute accuracy
        model_acc = accuracy_score(y_true, y_pred)
        chance_acc = accuracy_score(y_true, y_chance_pred)

        print(f"Model Accuracy: {model_acc:.3f}")
        print(f"Chance Baseline Accuracy: {chance_acc:.3f}\n")

        print("Model Classification Report:")
        print(classification_report(y_true, y_pred))

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
            # Fallback: use overall accuracy comparison (no statistical test)
            print(f"\nNote: Per-fold accuracies not provided. Cannot perform paired t-test.")
            if model_acc > chance_acc:
                print("The classifier has higher accuracy than chance baseline.")
            elif model_acc < chance_acc:
                print("The classifier has lower accuracy than chance baseline.")
            else:
                print("The classifier has the same accuracy as chance baseline.")
            
    def feature_importance(self, top_n=5):
        if not hasattr(self.model, "coef_"):
            print("Model is not trained yet.")
            return

        coefs = self.model.coef_

        if coefs.shape[0] == 1:
            # Binary classification
            importance = coefs[0]
            coef_df = pd.DataFrame({
                "feature": self.feature_columns,
                "coefficient": importance,
                "abs_coef": np.abs(importance)
            }).sort_values("abs_coef", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=coef_df.head(top_n), x="coefficient", y="feature", palette="coolwarm")
            plt.title("Top Feature Importances (Binary Logistic Regression)")
            plt.tight_layout()
            plt.show()

        else:
            if self.plot_flag:
                n_classes = coefs.shape[0]
                fig, axes = plt.subplots(nrows=n_classes, figsize=(8, 1.5 * n_classes), sharex=True)

                if n_classes == 1:
                    axes = [axes]

                for i, class_label in enumerate(self.model.classes_):
                    importance = coefs[i]
                    coef_df = pd.DataFrame({
                        "feature": self.feature_columns,
                        "coefficient": importance,
                        "abs_coef": np.abs(importance)
                    }).sort_values("abs_coef", ascending=False)

                    sns.barplot(
                        data=coef_df.head(top_n),
                        x="coefficient",
                        y="feature",
                        ax=axes[i],
                        # palette="coolwarm",
                    )
                    axes[i].set_ylabel("")
                    axes[i].set_title(f"Feature Importance for '{class_label}'")

                plt.tight_layout()
                plt.savefig(f'{self._fig_dir}/{self.clf}_{self.algo}_{self.reg}_feats_importance{self._fig_ext}')
                plt.show()

    def explain_with_shap(self):
        """Compute and plot SHAP values for multiclass Logistic Regression."""

        print("Generating SHAP explanations...")

        explainer = shap.Explainer(self.model, self.X, feature_names=self.feature_columns)
        shap_values = explainer(self.X) 

        if isinstance(shap_values, list) or len(shap_values.shape) == 3:
            # Multiclass case: loop through classes
            n_classes = len(self.model.classes_)

            if self.plot_flag:
                # plt.figure(figsize=(6.5, 2 * n_classes))
                plt.figure()

                for i, class_label in enumerate(self.model.classes_):

                    # print(f"Generating beeswarm for class '{class_label}'...")
                    ax = plt.subplot(n_classes, 1, i + 1)
                    shap.plots.beeswarm(
                        shap_values[:, :, i],
                        max_display=5,
                        show=False,
                        alpha=0.5,
                        plot_size=(8, 6),
                        color_bar=False,  # Disable colorbar (right y-axis)
                        # plot_size=(6, 4.5)
                        # order=variance_order
                    )
                    ax.set_xlabel(f"SHAP values for '{class_label}'", fontsize=10)
        
                    # ax.set_title(f"SHAP values for '{class_label}'", fontsize=10)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(10)  

                    for label in ax.get_xticklabels():
                        label.set_fontsize(10)  # Set your preferred fontsize here


                plt.tight_layout()
                plt.savefig(f'{self._fig_dir}/{self.clf}_{self.algo}_{self.reg}_shap{self._fig_ext}')
                plt.show()

        else:
            # Binary classification case
            shap.plots.beeswarm(shap_values, max_display=20)
            plt.title("SHAP Feature Impact Summary")
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify psilocybin lists")
    parser.add_argument('-clf', 
                        type=str,
                        choices=['playlist', 'phase'],
                        default='phase',
                        help='Select labels to choose for classification.')
    parser.add_argument('-reg', 
                        type=str,
                        choices=['log', 'rf'],
                        default='log',
                        help='Select the classification model to use.')
    parser.add_argument('-mean', 
                        dest='mean',
                        action='store_true',
                        default=False,
                        help='Select mean (song-level) features for classification.')    
    parser.add_argument('-pca', 
                        dest='pca',
                        action='store_false',
                        default=True,
                        help='Select to NOT perform PCA reduction.')
    parser.add_argument('-cv', 
                        type=int,
                        choices=[5, 10],
                        default=5,
                        help='Select a number of cross-validation splits')
    parser.add_argument('-plot', 
                        dest='plot',
                        action='store_true',
                        default=False,
                        help='Select to create plots.')
    parser.add_argument('-playlist', 
                        type=str,
                        choices=['all', 'chacruna_baldwin', 'chacruna_kelan_thomas2', 'compass_v2',
                                 'copenhagen', 'imperial1', 'imperial2', 'jh_classical', 'jh_overtone', 'most'],
                        default='all',
                        help='Select the playlist you want to download.')
    parser.add_argument('-resample', 
                        type=str,
                        choices=['smote', 'under'],
                        default=None,
                        help='Apply resampling to handle class imbalance: smote (oversample minority) or undersample (undersample majority).')
    parser.add_argument('-clip', 
                        dest='clip',
                        action='store_true',
                        default=False,
                        help='Clip the data of return phase.')
    parser.add_argument('-boundaries', 
                        dest='boundaries',
                        action='store_true',
                        default=False,
                        help='Remove ')                        
    args = parser.parse_args()

    if args.boundaries and args.clip:
        print('Cannot clip and remove boundaries at the same time!')
        exit()

    mean_str = '_mean' if args.mean else ''

    clf = PsiloClassifier(clf=args.clf, 
                          algo='compare_lld', 
                          mean=mean_str, 
                          cv=args.cv, 
                          plot_flag=args.plot, 
                          playlist=args.playlist, 
                          reg=args.reg,
                          resample_method=args.resample,
                          clip=args.clip,
                          boundaries=args.boundaries,
                          do_pca=args.pca)   