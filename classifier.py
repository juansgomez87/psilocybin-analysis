import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in matmul') 
warnings.filterwarnings('ignore', message='divide by zero encountered in matmul')
warnings.filterwarnings('ignore', message='overflow encountered in matmul') 
warnings.filterwarnings('ignore', message='invalid value encountered in divide') 

import pdb

class PsiloClassifier():
    def __init__(self, clf, algo, mean, cv, plot_flag, playlist, reg, resample_method=None):
        self.seed = 1987
        self.clf = clf
        self.algo = algo
        self.df = pd.read_csv(f'data/df_{algo}{mean}.csv', index_col=0)
        self.plot_flag = plot_flag
        self.playlist = playlist
        self.reg = reg  # log - logistic regression, rf - random forest
        self.resample_method = resample_method
        
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
        

        print('Doing PCA reduction!')
        self.pca_reduction()

        self.y = self.df[clf].values 
        self.cv = cv

        # Initialize model without class weights (will be set per fold)
        if self.reg == 'log':
            self.model = LogisticRegression(
                max_iter=2000,
                solver='lbfgs'
            )
        elif self.reg == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,  # or set to limit complexity
                random_state=self.seed,
                n_jobs=-1
            )
        
        self.run_classification()
        
    def pca_reduction(self):
        print('Calculating PCA, this might take time...')
        self.X = self.df[self.feature_columns].values
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
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
        
        for train_idx, test_idx in skf.split(self.X, self.y, groups=self.df['file'].values):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Apply resampling to training data only if specified
            if self.resample_method:
                print(f"Original training class distribution: {Counter(y_train)}")
                X_train, y_train, class_weight_dict = self._resample_fold(X_train, y_train)
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
            y_score = self.model.predict_proba(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_score_all.extend(y_score)
        
        # Convert lists to numpy arrays
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_score_all = np.array(y_score_all)
        
        # self.evaluate(y_true_all, y_pred_all, y_score_all)
        self.plot_stratified_splits()
        self.plot_roc_auc(y_true_all, y_score_all, self.model.classes_)
        self.compare_with_chance(y_true_all, y_pred_all)
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
            plt.savefig(f'figs/{args.clf}_{self.algo}_{self.reg}_class_dist.pdf')
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
        plt.savefig(f'figs/{args.clf}_{self.algo}_{self.reg}_roc.pdf')
        plt.show()


    def compare_with_chance(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.model.classes_)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 1)

        print('-'*50)
        print('Confusion Matrix:')
        print(f'{self.model.classes_}')
        print(cm)
        print('-'*50)

        if self.plot_flag:
            fig, ax = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
            disp.plot(
                ax=ax,
                xticks_rotation=90,
                cmap='Blues',  
                include_values=True,     
                colorbar=False,
            )

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            for i in range(len(self.model.classes_)):
                for j in range(len(self.model.classes_)):
                    text = disp.text_[i, j]
                    text.set_fontsize(14)
                    if i == j:
                        text.set_weight('bold')  # bold diagonal
                        # text.set_color('black') 
            # for text in disp.text_.ravel():
            #     text.set_fontsize(4)

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel('Predicted label', fontsize=15)
            ax.set_ylabel('True label', fontsize=15)
            plt.tight_layout()
            plt.savefig(f'figs/{args.clf}_{self.algo}_{self.reg}_confmat.pdf')
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

        # print("Baseline (Chance) Classification Report:")
        # print(classification_report(y_true, y_chance_pred))

        # Perform McNemar’s test
        contingency_table = np.array([
            [(y_true == y_pred).sum(), (y_true != y_pred).sum()],
            [(y_true == y_chance_pred).sum(), (y_true != y_chance_pred).sum()]
        ])
        result = mcnemar(contingency_table, exact=True)
        print(f"\nMcNemar’s test p-value: {result.pvalue:.7f}")

        if result.pvalue < 0.05 and model_acc > chance_acc:
            print("The classifier is significantly better than chance! 🎉")
        elif result.pvalue < 0.05 and model_acc < chance_acc:
            print("The classifier is significantly worse than chance. 😡")
        else:
            print("The classifier is NOT significantly better than chance. 🤔")
            
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
            # Multiclass classification in subplots
            if self.plot_flag:
                n_classes = coefs.shape[0]
                fig, axes = plt.subplots(nrows=n_classes, figsize=(10, 2 * n_classes), sharex=True)

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
                    axes[i].set_title(f"Feature Importances for class '{class_label}'")

                plt.tight_layout()
                plt.savefig(f'figs/{args.clf}_{self.algo}_{self.reg}_feats_importance.pdf')
                plt.show()

    def explain_with_shap(self):
        """Compute and plot SHAP values for multiclass Logistic Regression."""

        print("Generating SHAP explanations...")

        # Use LinearExplainer for multiclass model
        explainer = shap.Explainer(self.model, self.X, feature_names=self.feature_columns)
        shap_values = explainer(self.X)  # This is a list of SHAP values, one array per class
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
                        plot_size=(6, 6)
                        # plot_size=(6, 4.5)
                    )
                    ax.set_xlabel("")
        
                    ax.set_title(f"SHAP values for '{class_label}'", fontsize=10)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(8)  

                    for label in ax.get_xticklabels():
                        label.set_fontsize(8)  # Set your preferred fontsize here

                plt.tight_layout()
                plt.savefig(f'figs/{args.clf}_{self.algo}_{self.reg}_shap.pdf')
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
                        help='Select the regression model to use.')
    parser.add_argument('-mean', 
                        dest='mean',
                        action='store_true',
                        default=False,
                        help='Select mean (song-level) features for classification.')
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
    args = parser.parse_args()

    mean_str = '_mean' if args.mean else ''

    clf = PsiloClassifier(clf=args.clf, 
                          algo='compare_lld', 
                          mean=mean_str, 
                          cv=args.cv, 
                          plot_flag=args.plot, 
                          playlist=args.playlist, 
                          reg=args.reg,
                          resample_method=args.resample)   