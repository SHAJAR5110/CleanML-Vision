"""Pair images with labels from CSV file.

Handles loading labels.csv from ZIP uploads and joining with image metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels_csv(session_dir: Path) -> pd.DataFrame | None:
    """Load labels.csv if it exists in the session directory.
    
    Args:
        session_dir: Session directory that may contain labels.csv
        
    Returns:
        DataFrame with labels or None if not found
    """
    labels_path = session_dir / "labels.csv"
    if not labels_path.exists():
        return None
    
    try:
        df_labels = pd.read_csv(labels_path)
        return df_labels
    except Exception:
        return None


def join_with_labels(df: pd.DataFrame, session_dir: Path, params: dict | None = None) -> tuple[pd.DataFrame, str, str]:
    """Join image metadata with labels from labels.csv.
    
    Args:
        df: Image metadata DataFrame
        session_dir: Session directory containing labels.csv
        params: Optional dict with 'on' (column to join on, default 'image_id')
        
    Returns:
        (new_df, code, message)
    """
    # Load labels.csv
    df_labels = load_labels_csv(session_dir)
    
    if df_labels is None:
        return df, "", "Error: labels.csv not found in session directory."
    
    # Get join column
    join_on = (params or {}).get("on", "image_id")
    
    if join_on not in df.columns:
        return df, "", f"Error: Column '{join_on}' not found in image metadata."
    
    if join_on not in df_labels.columns:
        return df, "", f"Error: Column '{join_on}' not found in labels.csv."
    
    # Perform left join (keep all images, add labels where available)
    df_joined = df.merge(df_labels, on=join_on, how="left")
    
    # Count matched and unmatched
    matched = df_joined[join_on].isin(df_labels[join_on]).sum()
    unmatched = len(df_joined) - matched
    
    code = f"""import pandas as pd

# Load labels
df_labels = pd.read_csv('labels.csv')

# Join with image metadata
df = df.merge(df_labels, on='{join_on}', how='left')
"""
    
    message = f"Joined {matched} images with labels from labels.csv"
    if unmatched > 0:
        message += f" · {unmatched} images without labels"
    message += "."
    
    return df_joined, code, message


def filter_by_label(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Filter images by label value.
    
    Args:
        df: Image metadata DataFrame (must have label column)
        session_dir: Session directory
        params: Dict with 'column' (label column name) and 'values' (list of values to keep)
        
    Returns:
        (new_df, code, message)
    """
    column = params["column"]
    values = params["values"]
    
    if column not in df.columns:
        return df, "", f"Error: Column '{column}' not found in metadata."
    
    # Filter by values
    df_filtered = df[df[column].isin(values)].reset_index(drop=True)
    
    before = len(df)
    after = len(df_filtered)
    removed = before - after
    
    code = f"""# Filter by label values
df = df[df['{column}'].isin({values})].reset_index(drop=True)
"""
    
    message = f"Filtered to {after} images with {column} in {values} · removed {removed} images."
    
    return df_filtered, code, message


def balance_by_label(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Balance dataset by label distribution.
    
    Args:
        df: Image metadata DataFrame (must have label column)
        session_dir: Session directory
        params: Dict with 'column' (label column) and 'strategy' ('undersample'|'oversample')
        
    Returns:
        (new_df, code, message)
    """
    column = params["column"]
    strategy = params.get("strategy", "undersample")
    
    if column not in df.columns:
        return df, "", f"Error: Column '{column}' not found in metadata."
    
    if strategy not in ("undersample", "oversample"):
        return df, "", f"Invalid strategy: {strategy}. Use 'undersample' or 'oversample'."
    
    # Get class counts
    class_counts = df[column].value_counts()
    
    if strategy == "undersample":
        # Undersample to minority class size
        min_count = class_counts.min()
        
        balanced_dfs = []
        for label in class_counts.index:
            label_df = df[df[column] == label]
            sampled = label_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        code = f"""# Undersample to balance classes
min_count = df['{column}'].value_counts().min()
balanced_dfs = []
for label in df['{column}'].unique():
    label_df = df[df['{column}'] == label]
    sampled = label_df.sample(n=min_count, random_state=42)
    balanced_dfs.append(sampled)
df = pd.concat(balanced_dfs, ignore_index=True)
"""
        
        message = f"Undersampled to {min_count} images per class · total {len(df_balanced)} images."
        
    else:  # oversample
        # Oversample to majority class size
        max_count = class_counts.max()
        
        balanced_dfs = []
        for label in class_counts.index:
            label_df = df[df[column] == label]
            if len(label_df) < max_count:
                # Oversample with replacement
                sampled = label_df.sample(n=max_count, replace=True, random_state=42)
            else:
                sampled = label_df
            balanced_dfs.append(sampled)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        code = f"""# Oversample to balance classes
max_count = df['{column}'].value_counts().max()
balanced_dfs = []
for label in df['{column}'].unique():
    label_df = df[df['{column}'] == label]
    if len(label_df) < max_count:
        sampled = label_df.sample(n=max_count, replace=True, random_state=42)
    else:
        sampled = label_df
    balanced_dfs.append(sampled)
df = pd.concat(balanced_dfs, ignore_index=True)
"""
        
        message = f"Oversampled to {max_count} images per class · total {len(df_balanced)} images."
    
    return df_balanced, code, message


def split_by_label(df: pd.DataFrame, session_dir: Path, params: dict) -> tuple[pd.DataFrame, str, str]:
    """Split dataset into train/test by label (stratified).
    
    Args:
        df: Image metadata DataFrame (must have label column)
        session_dir: Session directory
        params: Dict with 'column' (label column), 'test_size' (0.0-1.0), 'random_state' (int)
        
    Returns:
        (new_df, code, message) - df gets 'split' column added
    """
    column = params["column"]
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))
    
    if column not in df.columns:
        return df, "", f"Error: Column '{column}' not found in metadata."
    
    if not (0.0 < test_size < 1.0):
        return df, "", f"Invalid test_size: {test_size}. Use range 0.0-1.0."
    
    df = df.copy()
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df[column],
        random_state=random_state
    )
    
    # Add split column
    df["split"] = "train"
    df.loc[test_indices, "split"] = "test"
    
    train_count = len(train_indices)
    test_count = len(test_indices)
    
    code = f"""from sklearn.model_selection import train_test_split

# Stratified split by {column}
train_indices, test_indices = train_test_split(
    df.index,
    test_size={test_size},
    stratify=df['{column}'],
    random_state={random_state}
)

df['split'] = 'train'
df.loc[test_indices, 'split'] = 'test'
"""
    
    message = f"Split dataset: {train_count} train, {test_count} test (stratified by {column})."
    
    return df, code, message


STRATEGIES = {
    "join_labels": join_with_labels,
    "filter_by_label": filter_by_label,
    "balance": balance_by_label,
    "split": split_by_label,
}


def apply(df: pd.DataFrame, column: str | None, strategy: str, params: dict | None = None, session_dir: Path | None = None) -> tuple[pd.DataFrame, str, str]:
    """Apply a pairing/labeling strategy.
    
    Args:
        df: Metadata DataFrame
        column: Not used (kept for consistency with tabular API)
        strategy: Strategy name from STRATEGIES
        params: Strategy-specific parameters
        session_dir: Optional session directory path (for testing)
        
    Returns:
        (new_df, code, message)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown pair strategy: {strategy}")
    
    # Extract session_dir from first image path
    if df.empty:
        return df, "", "No images to process."
    
    if session_dir is None:
        session_dir = Path.cwd()  # Will be overridden by caller in production
    
    return STRATEGIES[strategy](df, session_dir, params or {})
