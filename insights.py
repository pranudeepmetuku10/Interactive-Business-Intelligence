"""
Insights Generation Module for BI Dashboard
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class InsightType(Enum):
    TOP_PERFORMERS = "top_performers"
    BOTTOM_PERFORMERS = "bottom_performers"
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    DATA_QUALITY = "data_quality"
    DISTRIBUTION = "distribution"


class InsightPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    insight_type: InsightType
    title: str
    description: str
    priority: InsightPriority
    data: Optional[Dict[str, Any]] = None
    
    def to_markdown(self) -> str:
        priority_config = {
            InsightPriority.HIGH: {"icon": "🔴", "label": "HIGH PRIORITY", "color": "#e74c3c"},
            InsightPriority.MEDIUM: {"icon": "🟡", "label": "MEDIUM", "color": "#f39c12"},
            InsightPriority.LOW: {"icon": "🟢", "label": "LOW", "color": "#27ae60"}
        }
        config = priority_config.get(self.priority, priority_config[InsightPriority.MEDIUM])
        
        # Format description with better bullet points
        formatted_desc = self.description.replace("•", "\n>  📌")
        if not formatted_desc.startswith("\n"):
            formatted_desc = f"\n> {formatted_desc}"
        
        return f"""
---

<center>

### {config['icon']} {self.title}

**`{config['label']}`**

</center>

{formatted_desc}

"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs_val >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs_val >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    types = {'numerical': [], 'categorical': [], 'datetime': [], 'boolean': []}
    if df is None:
        return types
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            types['boolean'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types['datetime'].append(col)
        else:
            types['categorical'].append(col)
    return types


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)


# =============================================================================
# INSIGHT STRATEGIES
# =============================================================================

class InsightStrategy(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame, **kwargs) -> List[Insight]:
        pass
    @abstractmethod
    def get_name(self) -> str:
        pass


class TopBottomPerformersStrategy(InsightStrategy):
    def generate(self, df: pd.DataFrame, value_column: str = None, 
                 category_column: str = None, n: int = 5) -> List[Insight]:
        insights = []
        if df is None or df.empty:
            return insights
        
        col_types = get_column_types(df)
        if value_column is None and col_types['numerical']:
            value_column = col_types['numerical'][0]
        if value_column is None or value_column not in df.columns:
            return insights
        
        try:
            if category_column and category_column in df.columns:
                agg_df = df.groupby(category_column)[value_column].sum().reset_index()
                label_col = category_column
            else:
                agg_df = df[[value_column]].copy()
                agg_df['index'] = agg_df.index
                label_col = 'index'
            
            # Top performers
            top_n = agg_df.nlargest(n, value_column)
            if len(top_n) > 0:
                items = []
                for rank, (_, row) in enumerate(top_n.iterrows(), 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    items.append(f"| {medal} | **{str(row[label_col])[:25]}** | `{format_number(row[value_column])}` |")
                
                table = "| Rank | Name | Value |\n|:---:|:---|---:|\n" + "\n".join(items)
                insights.append(Insight(
                    InsightType.TOP_PERFORMERS, f"🏆 Top {len(top_n)} Performers by {value_column}",
                    f"**Highest performing items:**\n\n{table}", InsightPriority.HIGH
                ))
            
            # Bottom performers
            bottom_n = agg_df.nsmallest(n, value_column)
            if len(bottom_n) > 0:
                items = []
                for rank, (_, row) in enumerate(bottom_n.iterrows(), 1):
                    items.append(f"| {rank} | **{str(row[label_col])[:25]}** | `{format_number(row[value_column])}` |")
                
                table = "| Rank | Name | Value |\n|:---:|:---|---:|\n" + "\n".join(items)
                insights.append(Insight(
                    InsightType.BOTTOM_PERFORMERS, f"📉 Bottom {len(bottom_n)} by {value_column}",
                    f"**Lowest performing items:**\n\n{table}", InsightPriority.MEDIUM
                ))
        except Exception:
            pass
        return insights
    
    def get_name(self) -> str:
        return "Top/Bottom Performers"


class TrendAnalysisStrategy(InsightStrategy):
    def generate(self, df: pd.DataFrame, date_column: str = None, 
                 value_column: str = None) -> List[Insight]:
        insights = []
        if df is None or df.empty:
            return insights
        
        col_types = get_column_types(df)
        if date_column is None and col_types['datetime']:
            date_column = col_types['datetime'][0]
        if value_column is None and col_types['numerical']:
            value_column = col_types['numerical'][0]
        if not date_column or not value_column:
            return insights
        if date_column not in df.columns or value_column not in df.columns:
            return insights
        
        try:
            ts_df = df[[date_column, value_column]].copy()
            ts_df[date_column] = pd.to_datetime(ts_df[date_column])
            ts_df = ts_df.dropna().sort_values(date_column)
            ts_df = ts_df.groupby(date_column)[value_column].sum().reset_index()
            
            if len(ts_df) < 3:
                return insights
            
            values = ts_df[value_column].values
            slope, _ = np.polyfit(np.arange(len(values)), values, 1)
            
            start_val, end_val = values[0], values[-1]
            pct_change = (end_val - start_val) / abs(start_val) if start_val != 0 else 0
            
            direction = "UPWARD 📈" if slope > 0 else "DOWNWARD 📉"
            trend_emoji = "📈" if slope > 0 else "📉"
            
            mid = len(values) // 2
            period_change = (np.mean(values[mid:]) - np.mean(values[:mid])) / abs(np.mean(values[:mid])) if np.mean(values[:mid]) != 0 else 0
            
            priority = InsightPriority.HIGH if abs(pct_change) > 0.2 else InsightPriority.MEDIUM
            
            trend_table = f"""
| Metric | Value |
|:-------|------:|
| 📊 **Trend Direction** | **{direction}** |
| 📈 **Overall Change** | `{format_percentage(pct_change)}` |
| 🔄 **Recent vs Earlier** | `{format_percentage(period_change)}` |
| 🏁 **Starting Value** | `{format_number(start_val)}` |
| 🎯 **Ending Value** | `{format_number(end_val)}` |
"""
            
            insights.append(Insight(
                InsightType.TREND, f"{trend_emoji} Trend Analysis: {value_column}",
                trend_table,
                priority
            ))
        except Exception:
            pass
        return insights
    
    def get_name(self) -> str:
        return "Trend Analysis"


class AnomalyDetectionStrategy(InsightStrategy):
    def generate(self, df: pd.DataFrame, columns: List[str] = None) -> List[Insight]:
        insights = []
        if df is None or df.empty:
            return insights
        
        col_types = get_column_types(df)
        if columns is None:
            columns = col_types['numerical'][:5]
        
        for col in columns:
            if col not in df.columns:
                continue
            try:
                series = df[col].dropna()
                if len(series) < 10:
                    continue
                
                outliers = detect_outliers_iqr(series)
                count = outliers.sum()
                pct = count / len(series)
                
                if count > 0:
                    outlier_vals = series[outliers]
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    priority = InsightPriority.HIGH if pct > 0.05 else InsightPriority.MEDIUM
                    
                    severity = "🔴 **High**" if pct > 0.05 else "🟡 **Medium**" if pct > 0.02 else "🟢 **Low**"
                    
                    anomaly_table = f"""
| Metric | Value |
|:-------|------:|
| 🔢 **Outliers Found** | **{count}** ({format_percentage(pct)}) |
| ⚡ **Severity** | {severity} |
| ✅ **Normal Range** | `{format_number(q1)}` — `{format_number(q3)}` |
| ⚠️ **Outlier Range** | `{format_number(outlier_vals.min())}` — `{format_number(outlier_vals.max())}` |
"""
                    
                    insights.append(Insight(
                        InsightType.ANOMALY, f"⚠️ Outliers Detected: {col}",
                        anomaly_table,
                        priority
                    ))
            except Exception:
                continue
        return insights
    
    def get_name(self) -> str:
        return "Anomaly Detection"


class CorrelationInsightStrategy(InsightStrategy):
    def generate(self, df: pd.DataFrame, threshold: float = 0.7) -> List[Insight]:
        insights = []
        if df is None or df.empty:
            return insights
        
        num_cols = get_column_types(df)['numerical']
        if len(num_cols) < 2:
            return insights
        
        try:
            corr = df[num_cols].corr()
            strong = []
            
            for i, c1 in enumerate(num_cols):
                for j, c2 in enumerate(num_cols):
                    if i < j and abs(corr.loc[c1, c2]) >= threshold:
                        strong.append({'col1': c1, 'col2': c2, 'corr': corr.loc[c1, c2]})
            
            if strong:
                strong.sort(key=lambda x: abs(x['corr']), reverse=True)
                
                rows = []
                for c in strong[:5]:
                    direction = "🟢 Positive" if c['corr'] > 0 else "🔴 Negative"
                    strength = abs(c['corr'])
                    bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                    rows.append(f"| **{c['col1'][:15]}** | **{c['col2'][:15]}** | `{c['corr']:.3f}` | {direction} | {bar} |")
                
                table = "| Variable 1 | Variable 2 | Correlation | Direction | Strength |\n|:---|:---|:---:|:---:|:---:|\n" + "\n".join(rows)
                
                insights.append(Insight(
                    InsightType.CORRELATION, "🔗 Strong Correlations Found",
                    f"**Variables with correlation ≥ {threshold}:**\n\n{table}",
                    InsightPriority.MEDIUM
                ))
        except Exception:
            pass
        return insights
    
    def get_name(self) -> str:
        return "Correlation Analysis"


class DataQualityStrategy(InsightStrategy):
    def generate(self, df: pd.DataFrame) -> List[Insight]:
        insights = []
        if df is None or df.empty:
            return insights
        
        total = len(df)
        total_cells = total * len(df.columns)
        
        # Missing values
        missing = df.isnull().sum()
        cols_missing = [{'col': c, 'count': int(missing[c]), 'pct': missing[c]/total} 
                       for c in df.columns if missing[c] > 0]
        
        if cols_missing:
            cols_missing.sort(key=lambda x: x['pct'], reverse=True)
            total_missing = sum(c['count'] for c in cols_missing)
            completeness = 1 - total_missing / total_cells
            priority = InsightPriority.HIGH if completeness < 0.9 else InsightPriority.MEDIUM
            
            rows = []
            for c in cols_missing[:5]:
                pct_bar = "█" * int(c['pct'] * 20) + "░" * (20 - int(c['pct'] * 20))
                rows.append(f"| **{c['col'][:20]}** | `{c['count']:,}` | `{format_percentage(c['pct'])}` | {pct_bar} |")
            
            table = "| Column | Missing | Percent | Visual |\n|:---|---:|---:|:---|\n" + "\n".join(rows)
            
            status_emoji = "🔴" if completeness < 0.9 else "🟡" if completeness < 0.95 else "🟢"
            
            insights.append(Insight(
                InsightType.DATA_QUALITY, "📊 Data Quality: Missing Values",
                f"**{len(cols_missing)} columns** have missing data:\n\n{table}\n\n{status_emoji} **Data Completeness: `{format_percentage(completeness)}`**",
                priority
            ))
        
        # Duplicates
        dups = df.duplicated().sum()
        if dups > 0:
            pct = dups / total
            status = "🔴 **Critical**" if pct > 0.1 else "🟡 **Warning**" if pct > 0.05 else "🟢 **Minor**"
            insights.append(Insight(
                InsightType.DATA_QUALITY, "🔄 Duplicate Rows Detected",
                f"| Metric | Value |\n|:---|---:|\n| **Duplicates Found** | `{dups:,}` |\n| **Percentage** | `{format_percentage(pct)}` |\n| **Severity** | {status} |",
                InsightPriority.MEDIUM if pct > 0.05 else InsightPriority.LOW
            ))
        
        return insights
    
    def get_name(self) -> str:
        return "Data Quality"


# =============================================================================
# INSIGHT ENGINE
# =============================================================================

class InsightEngine:
    """Main engine for generating insights."""
    
    def __init__(self):
        self.strategies = [
            TopBottomPerformersStrategy(),
            TrendAnalysisStrategy(),
            AnomalyDetectionStrategy(),
            CorrelationInsightStrategy(),
            DataQualityStrategy()
        ]
    
    def generate_all_insights(self, df: pd.DataFrame, value_column: str = None,
                             category_column: str = None, date_column: str = None) -> List[Insight]:
        """Generate all insights from data."""
        all_insights = []
        
        for strategy in self.strategies:
            try:
                if isinstance(strategy, TopBottomPerformersStrategy):
                    insights = strategy.generate(df, value_column=value_column, category_column=category_column)
                elif isinstance(strategy, TrendAnalysisStrategy):
                    insights = strategy.generate(df, date_column=date_column, value_column=value_column)
                else:
                    insights = strategy.generate(df)
                all_insights.extend(insights)
            except Exception:
                continue
        
        # Sort by priority
        order = {InsightPriority.HIGH: 0, InsightPriority.MEDIUM: 1, InsightPriority.LOW: 2}
        all_insights.sort(key=lambda x: order.get(x.priority, 3))
        
        return all_insights
    
    def insights_to_markdown(self, insights: List[Insight]) -> str:
        """Convert insights to markdown."""
        if not insights:
            return """
<center>

## 💡 No Insights Generated

*Upload data and select columns to generate automated insights.*

</center>
"""
        
        header = f"""
<center>

# 💡 Business Intelligence Insights

### Discovered **{len(insights)}** key findings from your data

</center>

---
"""
        
        # Group insights by priority
        high_priority = [i for i in insights if i.priority == InsightPriority.HIGH]
        medium_priority = [i for i in insights if i.priority == InsightPriority.MEDIUM]
        low_priority = [i for i in insights if i.priority == InsightPriority.LOW]
        
        sections = [header]
        
        if high_priority:
            sections.append("\n## 🔴 Critical Insights\n")
            sections.extend([i.to_markdown() for i in high_priority])
        
        if medium_priority:
            sections.append("\n## 🟡 Important Findings\n")
            sections.extend([i.to_markdown() for i in medium_priority])
        
        if low_priority:
            sections.append("\n## 🟢 Additional Observations\n")
            sections.extend([i.to_markdown() for i in low_priority])
        
        return "\n".join(sections)
    
    def get_strategy_names(self) -> List[str]:
        """Get names of all strategies."""
        return [s.get_name() for s in self.strategies]