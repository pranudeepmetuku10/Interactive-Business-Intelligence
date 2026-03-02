"""
Utility Module for BI Dashboard
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

SUPPORTED_FORMATS = {
    'csv': ['.csv'],
    'excel': ['.xlsx', '.xls']
}

MAX_PREVIEW_ROWS = 100
DEFAULT_PREVIEW_ROWS = 10

AGGREGATION_METHODS = ['sum', 'mean', 'median', 'count', 'min', 'max', 'std']

TIME_FREQUENCIES = {
    'D': 'Daily',
    'W': 'Weekly', 
    'M': 'Monthly',
    'Q': 'Quarterly',
    'Y': 'Yearly'
}

COLOR_PALETTE = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
    '#95C623', '#5C4D7D', '#E84855', '#2D3047', '#93B7BE'
]

CHART_TYPES = {
    'time_series': 'Time Series Plot',
    'distribution': 'Distribution Plot',
    'category': 'Category Analysis',
    'scatter': 'Scatter Plot',
    'correlation': 'Correlation Heatmap',
    'box': 'Box Plot'
}



# TYPE DETECTION FUNCTIONS

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize DataFrame columns by their data types."""
    column_types = {
        'numerical': [],
        'categorical': [],
        'datetime': [],
        'boolean': []
    }
    
    if df is None or df.empty:
        return column_types
    
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            column_types['boolean'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        else:
            if is_potential_datetime(df[col]):
                column_types['datetime'].append(col)
            else:
                column_types['categorical'].append(col)
    
    return column_types


def is_potential_datetime(series: pd.Series, sample_size: int = 100) -> bool:
    """Check if a series could be parsed as datetime."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    
    sample = non_null.head(sample_size)
    try:
        pd.to_datetime(sample, format='mixed')
        return True
    except (ValueError, TypeError, Exception):
        return False


def is_numeric_column(series: pd.Series) -> bool:
    """Check if a series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_column(series: pd.Series, threshold: float = 0.5) -> bool:
    """Check if a series should be treated as categorical."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
    return unique_ratio <= threshold


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with K/M/B suffixes."""
    if pd.isna(value):
        return "N/A"
    
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"{value / 1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as percentage."""
    if pd.isna(value):
        return "N/A"
    if -1 <= value <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_value) < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} PB"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate a string with ellipsis."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate a DataFrame for basic requirements."""
    if df is None:
        return False, "No data provided"
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a DataFrame"
    if df.empty:
        return False, "DataFrame is empty"
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    return True, "Data is valid"


def check_column_exists(df: pd.DataFrame, column: str) -> bool:
    """Check if a column exists in the DataFrame."""
    if df is None:
        return False
    return column in df.columns


def check_columns_exist(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
    """Check if multiple columns exist."""
    if df is None:
        return False, columns
    missing = [col for col in columns if col not in df.columns]
    return len(missing) == 0, missing


def get_memory_usage(df: pd.DataFrame) -> str:
    """Get memory usage in human-readable format."""
    if df is None:
        return "N/A"
    bytes_used = df.memory_usage(deep=True).sum()
    return format_bytes(bytes_used)


# =============================================================================
# STATISTICAL HELPER FUNCTIONS
# =============================================================================

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = (series - mean) / std
    return abs(z_scores) > threshold


def calculate_percentile_rank(series: pd.Series) -> pd.Series:
    """Calculate percentile rank (0-100)."""
    return series.rank(pct=True) * 100


def calculate_summary_stats(series: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive summary statistics."""
    series = series.dropna()
    if len(series) == 0:
        return {}
    
    return {
        'count': len(series),
        'mean': float(series.mean()),
        'std': float(series.std()) if len(series) > 1 else 0.0,
        'min': float(series.min()),
        'q1': float(series.quantile(0.25)),
        'median': float(series.median()),
        'q3': float(series.quantile(0.75)),
        'max': float(series.max()),
        'skewness': float(series.skew()) if len(series) > 2 else 0.0,
        'kurtosis': float(series.kurtosis()) if len(series) > 3 else 0.0
    }


# =============================================================================
# DATETIME HELPER FUNCTIONS
# =============================================================================

def get_date_range(series: pd.Series) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get min/max dates from a series."""
    try:
        dt_series = pd.to_datetime(series)
        return dt_series.min(), dt_series.max()
    except (ValueError, TypeError):
        return None, None


def extract_date_components(df: pd.DataFrame, date_column: str, 
                           components: List[str] = None) -> pd.DataFrame:
    """Extract date components from a datetime column."""
    if components is None:
        components = ['year', 'month', 'day', 'dayofweek']
    
    df = df.copy()
    if date_column not in df.columns:
        return df
    
    try:
        dt_series = pd.to_datetime(df[date_column])
        component_map = {
            'year': dt_series.dt.year,
            'month': dt_series.dt.month,
            'day': dt_series.dt.day,
            'dayofweek': dt_series.dt.dayofweek,
            'quarter': dt_series.dt.quarter,
            'hour': dt_series.dt.hour,
            'week': dt_series.dt.isocalendar().week
        }
        for comp in components:
            if comp in component_map:
                df[f'{date_column}_{comp}'] = component_map[comp]
    except (ValueError, TypeError):
        pass
    
    return df


# =============================================================================
# FILE & EXPORT HELPERS
# =============================================================================

def generate_filename(prefix: str, extension: str) -> str:
    """Generate a filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def get_export_filename(base_name: str = "data", format: str = "csv") -> str:
    """Generate an export filename."""
    return generate_filename(base_name, format)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert DataFrame to markdown table."""
    if df is None or df.empty:
        return "*No data available*"
    
    if len(df) > max_rows:
        display_df = df.head(max_rows)
        footer = f"\n\n*Showing {max_rows} of {len(df):,} rows*"
    else:
        display_df = df
        footer = ""
    
    return display_df.to_markdown(index=False) + footer


# =============================================================================
# MISCELLANEOUS HELPERS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)