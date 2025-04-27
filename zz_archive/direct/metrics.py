
import pandas as pd, numpy as np, itertools, collections, math
from typing import Dict, List, Tuple
from scipy.stats import chi2_contingency, entropy, pointbiserialr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def category_frequencies(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    cnt = {c: df['category_sequence'].explode().value_counts().get(c,0) for c in categories}
    return pd.Series(cnt).to_frame('count')

def transition_matrix(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    pairs = []
    for seq in df['category_sequence']:
        pairs.extend(zip(seq[:-1], seq[1:]))
    mat = pd.DataFrame(0, index=categories, columns=categories, dtype=int)
    for a,b in pairs:
        mat.loc[a,b] +=1
    return mat

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5*(p+q)
    return 0.5*(entropy(p, m, base=2)+ entropy(q, m, base=2))

def compute_entropy(seq: List[str]) -> float:
    counts = collections.Counter(seq)
    probs = np.array(list(counts.values()))/len(seq)
    return entropy(probs, base=2)

def lexical_yules_k(text: str) -> float:
    tokens = text.split()
    freq = collections.Counter(tokens).values()
    m1 = sum(freq)
    m2 = sum(f*f for f in freq)
    return 1e4*(m2 - m1)/(m1*m1)

def backtracking_correlation(df: pd.DataFrame, accuracy_col: str='is_correct') -> Tuple[float,float]:
    from scipy.stats import pointbiserialr
    backtrack = df['category_sequence'].apply(lambda s: int('backtracking_revision' in s))
    return pointbiserialr(backtrack, df[accuracy_col])

def xtp_train_test(df: pd.DataFrame, accuracy_col: str='is_correct', test_size: float=0.3, random_state: int=0):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['category_sequence'])
    y = df[accuracy_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    return auc, fpr, tpr
