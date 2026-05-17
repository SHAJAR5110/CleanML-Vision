"""Text-column cleaning."""

from __future__ import annotations

import pandas as pd

# Compact English stopwords list (no NLTK dependency).
STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "she", "so", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "do", "does", "did", "had", "having", "been",
    "being", "am", "could", "should", "shall", "may", "might", "must",
    "not", "no", "nor", "only", "own", "same", "than", "too", "very",
    "can", "just", "more", "most", "some", "such", "any", "all", "each",
    "few", "other", "out", "over", "under", "up", "down", "off",
}


def _as_string(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype("string")


def strip(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.strip()
    code = f"df['{column}'] = df['{column}'].astype('string').str.strip()"
    return df, code, f"Stripped whitespace in '{column}'."


def lowercase(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.lower()
    code = f"df['{column}'] = df['{column}'].astype('string').str.lower()"
    return df, code, f"Lowercased '{column}'."


def collapse_spaces(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.replace(r"\s+", " ", regex=True).str.strip()
    code = f"df['{column}'] = df['{column}'].astype('string').str.replace(r'\\s+', ' ', regex=True).str.strip()"
    return df, code, f"Collapsed whitespace in '{column}'."


def remove_special_chars(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
    code = f"df['{column}'] = df['{column}'].astype('string').str.replace(r'[^A-Za-z0-9\\s]', '', regex=True)"
    return df, code, f"Removed special characters in '{column}'."


def remove_punctuation(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.replace(r"[^\w\s]", "", regex=True)
    code = f"df['{column}'] = df['{column}'].astype('string').str.replace(r'[^\\w\\s]', '', regex=True)"
    return df, code, f"Removed punctuation in '{column}'."


def remove_stopwords(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    custom = (params or {}).get("words")
    stopset = set(custom) if custom else STOPWORDS_EN

    def _strip(text):
        if text is None or pd.isna(text):
            return text
        tokens = str(text).split()
        return " ".join(t for t in tokens if t.lower() not in stopset)

    df[column] = df[column].map(_strip)
    code = (
        "_stop = " + repr(sorted(stopset)) + "\n"
        f"df['{column}'] = df['{column}'].map(lambda t: ' '.join(w for w in str(t).split() if w.lower() not in _stop) if pd.notna(t) else t)"
    )
    return df, code, f"Removed English stopwords from '{column}'."


def word_count(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    new_col = f"{column}_word_count"
    df[new_col] = _as_string(df, column).fillna("").str.split().str.len()
    code = f"df['{new_col}'] = df['{column}'].astype('string').fillna('').str.split().str.len()"
    return df, code, f"Added word-count feature '{new_col}' from '{column}'."


def char_count(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    new_col = f"{column}_char_count"
    df[new_col] = _as_string(df, column).fillna("").str.len()
    code = f"df['{new_col}'] = df['{column}'].astype('string').fillna('').str.len()"
    return df, code, f"Added char-count feature '{new_col}' from '{column}'."


def alphabetic_only(df: pd.DataFrame, column: str, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df[column] = _as_string(df, column).str.replace(r"[^A-Za-z\s]", "", regex=True)
    code = f"df['{column}'] = df['{column}'].astype('string').str.replace(r'[^A-Za-z\\s]', '', regex=True)"
    return df, code, f"Kept only alphabetic characters in '{column}'."


STRATEGIES = {
    "strip": strip,
    "lowercase": lowercase,
    "collapse_spaces": collapse_spaces,
    "remove_special": remove_special_chars,
    "remove_punctuation": remove_punctuation,
    "remove_stopwords": remove_stopwords,
    "alphabetic_only": alphabetic_only,
    "word_count": word_count,
    "char_count": char_count,
}


def apply(df, column, strategy, params=None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown text strategy: {strategy}")
    return STRATEGIES[strategy](df, column, params or {})
