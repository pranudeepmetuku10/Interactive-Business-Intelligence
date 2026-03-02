"""
Data Processing Module for BI Dashboard
Handles data loading, cleaning, filtering, and profiling.
Implements Strategy Pattern for flexible filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize DataFrame columns by their data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with keys 'numerical', 'categorical', 'datetime', 'boolean'
    """
    column_types = {
        'numerical': [],
        'categorical': [],
        'datetime': [],
        'boolean': []
    }
    
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            column_types['boolean'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        else:
            # Check if it could be parsed as datetime
            if _is_potential_datetime(df[col]):
                column_types['datetime'].append(col)
            else:
                column_types['categorical'].append(col)
    
    return column_types


def _is_potential_datetime(series: pd.Series, sample_size: int = 100) -> bool:
    """Check if a series could be parsed as datetime."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    
    sample = non_null.head(sample_size)
    
    try:
        pd.to_datetime(sample, format='mixed')
        return True
    except (ValueError, TypeError):
        return False


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        series: Numerical series
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Boolean series indicating outliers
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (series < lower_bound) | (series > upper_bound)


# =============================================================================
# STRATEGY PATTERN: FILTER STRATEGIES
# =============================================================================

class FilterStrategy(ABC):
    """Abstract base class for filtering strategies."""
    
    @abstractmethod
    def apply(self, df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
        """
        Apply the filter to the DataFrame.
        
        Args:
            df: Input DataFrame
            column: Column to filter on
            value: Filter value(s)
            
        Returns:
            Filtered DataFrame
        """
        pass
    
    @abstractmethod
    def get_filter_options(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get available filter options for the column.
        
        Args:
            df: Input DataFrame
            column: Column name
            
        Returns:
            Dictionary with filter options
        """
        pass


class NumericalFilterStrategy(FilterStrategy):
    """Filter strategy for numerical columns using range selection."""
    
    def apply(self, df: pd.DataFrame, column: str, value: Tuple[float, float]) -> pd.DataFrame:
        """
        Apply numerical range filter.
        
        Args:
            df: Input DataFrame
            column: Column to filter
            value: Tuple of (min_value, max_value)
            
        Returns:
            Filtered DataFrame
        """
        if value is None or column not in df.columns:
            return df
        
        min_val, max_val = value
        return df[(df[column] >= min_val) & (df[column] <= max_val)]
    
    def get_filter_options(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get min, max, and statistics for numerical column."""
        series = df[column].dropna()
        
        if len(series) == 0:
            return {'min': 0, 'max': 1, 'step': 0.1}
        
        min_val = float(series.min())
        max_val = float(series.max())
        range_val = max_val - min_val
        
        # Calculate appropriate step
        if range_val == 0:
            step = 1
        elif range_val < 1:
            step = range_val / 100
        elif range_val < 100:
            step = 1
        else:
            step = range_val / 100
        
        return {
            'min': min_val,
            'max': max_val,
            'step': step,
            'mean': float(series.mean()),
            'median': float(series.median())
        }


class CategoricalFilterStrategy(FilterStrategy):
    """Filter strategy for categorical columns using multi-select."""
    
    def apply(self, df: pd.DataFrame, column: str, value: List[str]) -> pd.DataFrame:
        """
        Apply categorical filter.
        
        Args:
            df: Input DataFrame
            column: Column to filter
            value: List of selected categories
            
        Returns:
            Filtered DataFrame
        """
        if value is None or len(value) == 0 or column not in df.columns:
            return df
        
        return df[df[column].isin(value)]
    
    def get_filter_options(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get unique values and their counts."""
        value_counts = df[column].value_counts()
        
        return {
            'choices': list(df[column].dropna().unique()),
            'value_counts': value_counts.to_dict(),
            'num_unique': len(value_counts)
        }


class DateTimeFilterStrategy(FilterStrategy):
    """Filter strategy for datetime columns using date range."""
    
    def apply(self, df: pd.DataFrame, column: str, value: Tuple[str, str]) -> pd.DataFrame:
        """
        Apply datetime range filter.
        
        Args:
            df: Input DataFrame
            column: Column to filter
            value: Tuple of (start_date, end_date) as strings
            
        Returns:
            Filtered DataFrame
        """
        if value is None or column not in df.columns:
            return df
        
        start_date, end_date = value
        
        if not start_date and not end_date:
            return df
        
        try:
            result = df.copy()
            dt_column = pd.to_datetime(result[column])
            
            if start_date:
                result = result[dt_column >= pd.to_datetime(start_date)]
                dt_column = pd.to_datetime(result[column])
            
            if end_date:
                result = result[dt_column <= pd.to_datetime(end_date)]
            
            return result
            
        except (ValueError, TypeError):
            return df
    
    def get_filter_options(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get date range for datetime column."""
        try:
            dt_series = pd.to_datetime(df[column])
            min_date = dt_series.min()
            max_date = dt_series.max()
            
            return {
                'min_date': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None,
                'max_date': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
            }
        except (ValueError, TypeError):
            return {'min_date': None, 'max_date': None}


class BooleanFilterStrategy(FilterStrategy):
    """Filter strategy for boolean columns."""
    
    def apply(self, df: pd.DataFrame, column: str, value: Optional[bool]) -> pd.DataFrame:
        """
        Apply boolean filter.
        
        Args:
            df: Input DataFrame
            column: Column to filter
            value: Boolean value or None for all
            
        Returns:
            Filtered DataFrame
        """
        if value is None or column not in df.columns:
            return df
        
        return df[df[column] == value]
    
    def get_filter_options(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get boolean value counts."""
        value_counts = df[column].value_counts()
        
        return {
            'true_count': int(value_counts.get(True, 0)),
            'false_count': int(value_counts.get(False, 0))
        }


# =============================================================================
# FILTER FACTORY
# =============================================================================

class FilterFactory:
    """Factory class to create appropriate filter strategies."""
    
    _strategies = {
        'numerical': NumericalFilterStrategy,
        'categorical': CategoricalFilterStrategy,
        'datetime': DateTimeFilterStrategy,
        'boolean': BooleanFilterStrategy
    }
    
    @classmethod
    def get_strategy(cls, column_type: str) -> FilterStrategy:
        """
        Get the appropriate filter strategy for a column type.
        
        Args:
            column_type: Type of column ('numerical', 'categorical', 'datetime', 'boolean')
            
        Returns:
            Appropriate FilterStrategy instance
        """
        strategy_class = cls._strategies.get(column_type, CategoricalFilterStrategy)
        return strategy_class()


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Handles loading data from various file formats."""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls']
    }
    
    def load_file(self, file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load data from uploaded file.
        
        Args:
            file: Uploaded file object from Gradio
            
        Returns:
            Tuple of (DataFrame or None, status message)
        """
        if file is None:
            return None, "No file uploaded"
        
        try:
            file_path = file if isinstance(file, str) else file.name
            
            # Determine file type and load
            if file_path.endswith('.csv'):
                df = self._load_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = self._load_excel(file_path)
            else:
                return None, "Unsupported format. Please upload CSV or Excel files."
            
            # Auto-convert datetime columns
            df = self._infer_datetime_columns(df)
            
            # Validate
            if df.empty:
                return None, "The file is empty"
            
            return df, f"Loaded {len(df):,} rows and {len(df.columns)} columns"
            
        except pd.errors.EmptyDataError:
            return None, "The file is empty"
        except pd.errors.ParserError as e:
            return None, f"Error parsing file: {str(e)}"
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with encoding fallback."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path, engine='openpyxl')
    
    def _infer_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-convert potential datetime columns."""
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object' and _is_potential_datetime(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], format='mixed')
                except (ValueError, TypeError):
                    pass
        
        return df


# =============================================================================
# DATA PROFILER
# =============================================================================

class DataProfiler:
    """Generates comprehensive data profiles and statistics."""
    
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a complete data profile.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing profile information
        """
        if df is None or df.empty:
            return {}
        
        column_types = get_column_types(df)
        
        return {
            'overview': self._get_overview(df),
            'column_types': column_types,
            'numerical_stats': self._get_numerical_stats(df, column_types['numerical']),
            'categorical_stats': self._get_categorical_stats(df, column_types['categorical']),
            'datetime_stats': self._get_datetime_stats(df, column_types['datetime']),
            'missing_values': self._get_missing_values(df),
            'correlations': self._get_correlations(df, column_types['numerical'])
        }
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get general dataset overview."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_bytes': df.memory_usage(deep=True).sum(),
            'duplicated_rows': int(df.duplicated().sum()),
            'total_missing': int(df.isnull().sum().sum())
        }
    
    def _get_numerical_stats(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """Calculate statistics for numerical columns."""
        stats = {}
        
        for col in columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            stats[col] = {
                'count': int(len(series)),
                'mean': float(series.mean()),
                'std': float(series.std()) if len(series) > 1 else 0.0,
                'min': float(series.min()),
                'q1': float(series.quantile(0.25)),
                'median': float(series.median()),
                'q3': float(series.quantile(0.75)),
                'max': float(series.max()),
                'skewness': float(series.skew()) if len(series) > 2 else 0.0,
                'zeros': int((series == 0).sum()),
                'negatives': int((series < 0).sum()),
                'outliers': int(detect_outliers_iqr(series).sum())
            }
        
        return stats
    
    def _get_categorical_stats(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """Calculate statistics for categorical columns."""
        stats = {}
        
        for col in columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            mode_val = series.mode()
            
            stats[col] = {
                'count': int(len(series)),
                'unique': int(series.nunique()),
                'mode': str(mode_val.iloc[0]) if len(mode_val) > 0 else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5': value_counts.head(5).to_dict()
            }
        
        return stats
    
    def _get_datetime_stats(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """Calculate statistics for datetime columns."""
        stats = {}
        
        for col in columns:
            try:
                series = pd.to_datetime(df[col]).dropna()
                
                if len(series) == 0:
                    continue
                
                stats[col] = {
                    'count': int(len(series)),
                    'min': series.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'max': series.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'range_days': int((series.max() - series.min()).days)
                }
            except (ValueError, TypeError):
                continue
        
        return stats
    
    def _get_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze missing values in the dataset."""
        missing = df.isnull().sum()
        total = len(df)
        
        return {
            col: {
                'count': int(missing[col]),
                'percentage': round(missing[col] / total * 100, 2) if total > 0 else 0
            }
            for col in df.columns
        }
    
    def _get_correlations(self, df: pd.DataFrame, numerical_columns: List[str]) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for numerical columns."""
        if len(numerical_columns) < 2:
            return None
        
        try:
            return df[numerical_columns].corr()
        except Exception:
            return None


# =============================================================================
# DATA FILTER ENGINE
# =============================================================================

class DataFilterEngine:
    """Engine for applying multiple filters to data."""
    
    def __init__(self):
        self.active_filters: Dict[str, Tuple[str, Any]] = {}
    
    def add_filter(self, column: str, column_type: str, value: Any) -> None:
        """
        Add or update a filter.
        
        Args:
            column: Column name
            column_type: Type of column ('numerical', 'categorical', 'datetime', 'boolean')
            value: Filter value
        """
        self.active_filters[column] = (column_type, value)
    
    def remove_filter(self, column: str) -> None:
        """Remove a filter by column name."""
        if column in self.active_filters:
            del self.active_filters[column]
    
    def clear_filters(self) -> None:
        """Clear all active filters."""
        self.active_filters.clear()
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all active filters to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        if df is None:
            return None
        
        result = df.copy()
        
        for column, (column_type, value) in self.active_filters.items():
            strategy = FilterFactory.get_strategy(column_type)
            result = strategy.apply(result, column, value)
        
        return result
    
    def get_filter_summary(self) -> List[str]:
        """Get a summary of active filters."""
        summary = []
        
        for column, (column_type, value) in self.active_filters.items():
            if column_type == 'numerical':
                summary.append(f"{column}: {value[0]:.2f} to {value[1]:.2f}")
            elif column_type == 'categorical':
                summary.append(f"{column}: {len(value)} selected")
            elif column_type == 'datetime':
                summary.append(f"{column}: {value[0]} to {value[1]}")
            elif column_type == 'boolean':
                summary.append(f"{column}: {value}")
        
        return summary
    
    def get_active_filter_count(self) -> int:
        """Get number of active filters."""
        return len(self.active_filters)


# =============================================================================
# DATA AGGREGATOR
# =============================================================================

class DataAggregator:
    """Handles data aggregation operations."""
    
    SUPPORTED_METHODS = ['sum', 'mean', 'median', 'count', 'min', 'max', 'std']
    
    @classmethod
    def aggregate(
        cls,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        value_column: str,
        method: str = 'sum'
    ) -> pd.DataFrame:
        """
        Aggregate data by specified columns and method.
        
        Args:
            df: Input DataFrame
            group_by: Column(s) to group by
            value_column: Column to aggregate
            method: Aggregation method
            
        Returns:
            Aggregated DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Validate columns
        for col in group_by + [value_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
        
        if method not in cls.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Use: {cls.SUPPORTED_METHODS}")
        
        result = df.groupby(group_by, as_index=False)[value_column].agg(method)
        result.columns = list(group_by) + [f'{value_column}_{method}']
        
        return result
    
    @classmethod
    def time_series_aggregate(
        cls,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        freq: str = 'M',
        method: str = 'sum'
    ) -> pd.DataFrame:
        """
        Aggregate data by time periods.
        
        Args:
            df: Input DataFrame
            date_column: Datetime column
            value_column: Column to aggregate
            freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
            method: Aggregation method
            
        Returns:
            Time-aggregated DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        temp_df = df.copy()
        temp_df[date_column] = pd.to_datetime(temp_df[date_column])
        temp_df = temp_df.set_index(date_column)
        
        result = getattr(temp_df[value_column].resample(freq), method)()
        result = result.reset_index()
        result.columns = [date_column, f'{value_column}_{method}']
        
        return result