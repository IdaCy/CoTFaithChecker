
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_category_bars(freq_df: pd.DataFrame, title: str):
    freq_df.plot(kind='bar', legend=False)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.tight_layout()

def plot_transition_heatmap(mat: pd.DataFrame, title: str):
    sns.heatmap(mat, annot=False, cmap='Blues')
    plt.title(title)
    plt.ylabel('From')
    plt.xlabel('To')
    plt.tight_layout()

def plot_js_matrix(js_mat: pd.DataFrame, title: str):
    sns.heatmap(js_mat, annot=True, cmap='Reds')
    plt.title(title)
    plt.tight_layout()

def plot_length_entropy_scatter(lengths, entropies, title:str):
    plt.scatter(lengths, entropies)
    plt.xlabel('Chain length (#tokens)')
    plt.ylabel('Category entropy')
    plt.title(title)
    plt.tight_layout()

def plot_backtracking_accuracy(df: pd.DataFrame, title:str='Backtracking vs Accuracy'):
    sns.boxplot(x='backtracked', y='accuracy', data=df)
    plt.title(title)
    plt.tight_layout()

def plot_xtp_roc(fpr, tpr, auc, title:str='Explain‑then‑Predict ROC'):
    plt.plot(fpr, tpr, label=f'AUC={auc:0.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
