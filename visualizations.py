"""
Visualization Module for BI Dashboard 
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

# =============================================================================
# CONSTANTS
# =============================================================================

COLOR_PALETTE = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
    '#95C623', '#5C4D7D', '#E84855', '#2D3047', '#93B7BE'
]

CHART_TYPES = {
    'time_series': 'Time Series Plot',
    'distribution': 'Distribution Plot', 
    'category': 'Category Analysis',
    'scatter': 'Scatter Plot',
    'correlation': 'Correlation Heatmap'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, 
                       showarrow=False, font=dict(size=16))
    fig.update_layout(template='plotly_white', height=400)
    return fig

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns by data type."""
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

# =============================================================================
# VISUALIZATION STRATEGIES
# =============================================================================

class VisualizationStrategy(ABC):
    @abstractmethod
    def create(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        pass
    @abstractmethod
    def get_required_params(self) -> List[str]:
        pass


class TimeSeriesStrategy(VisualizationStrategy):
    """Time series visualization strategy."""
    
    def create(self, df: pd.DataFrame, date_column: str, value_column: str, 
               aggregation: str = 'sum', title: str = None) -> go.Figure:
        if df is None or df.empty:
            return create_empty_figure("No data available")
        if date_column not in df.columns or value_column not in df.columns:
            return create_empty_figure("Selected columns not found")
        
        try:
            plot_df = df[[date_column, value_column]].copy()
            plot_df[date_column] = pd.to_datetime(plot_df[date_column])
            plot_df = plot_df.dropna()
            
            agg_df = plot_df.groupby(date_column)[value_column].agg(aggregation).reset_index()
            agg_df = agg_df.sort_values(date_column)
            
            if agg_df.empty:
                return create_empty_figure("No data after aggregation")
            
            # Convert to lists
            x_data = agg_df[date_column].tolist()
            y_data = agg_df[value_column].tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                mode='lines+markers', name=value_column,
                line=dict(color=COLOR_PALETTE[0], width=2),
                marker=dict(size=6)
            ))
            
            # Trend line
            if len(agg_df) > 2:
                x_num = np.arange(len(agg_df))
                y_arr = np.array(y_data)
                z = np.polyfit(x_num, y_arr, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=x_data, y=p(x_num).tolist(),
                    mode='lines', name='Trend',
                    line=dict(color=COLOR_PALETTE[1], dash='dash')
                ))
            
            fig.update_layout(
                title=title or f'{value_column} Over Time ({aggregation})',
                xaxis_title=date_column, yaxis_title=value_column,
                template='plotly_white', height=500, hovermode='x unified'
            )
            return fig
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def get_required_params(self) -> List[str]:
        return ['date_column', 'value_column']


class DistributionStrategy(VisualizationStrategy):
    """Distribution visualization strategy."""
    
    def create(self, df: pd.DataFrame, column: str, plot_type: str = 'histogram',
               bins: int = 30, title: str = None) -> go.Figure:
        if df is None or df.empty:
            return create_empty_figure("No data available")
        if column not in df.columns:
            return create_empty_figure(f"Column '{column}' not found")
        
        try:
            data = df[column].dropna()
            if len(data) == 0:
                return create_empty_figure("No valid data")
            
            # Convert to list
            data_list = data.tolist()
            
            fig = go.Figure()
            
            if plot_type == 'histogram':
                fig.add_trace(go.Histogram(
                    x=data_list, nbinsx=bins,
                    marker_color=COLOR_PALETTE[0], opacity=0.75
                ))
                
                mean_val = float(data.mean())
                median_val = float(data.median())
                
                fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_PALETTE[1],
                             annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
                fig.add_vline(x=median_val, line_dash="dot", line_color=COLOR_PALETTE[2],
                             annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
                
                fig.update_layout(xaxis_title=column, yaxis_title='Count', bargap=0.05)
            else:
                fig.add_trace(go.Box(
                    y=data_list, name=column,
                    marker_color=COLOR_PALETTE[0], boxmean='sd'
                ))
                fig.update_layout(yaxis_title=column)
            
            fig.update_layout(
                title=title or f'Distribution of {column}',
                template='plotly_white', height=500
            )
            return fig
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def get_required_params(self) -> List[str]:
        return ['column']


class CategoryAnalysisStrategy(VisualizationStrategy):
    """Category analysis visualization strategy."""
    
    def create(self, df: pd.DataFrame, category_column: str, value_column: str = None,
               aggregation: str = 'count', plot_type: str = 'bar', top_n: int = 10,
               title: str = None) -> go.Figure:
        if df is None or df.empty:
            return create_empty_figure("No data available")
        if category_column not in df.columns:
            return create_empty_figure(f"Column '{category_column}' not found")
        
        try:
            if aggregation == 'count' or value_column is None:
                agg_df = df[category_column].value_counts().reset_index()
                agg_df.columns = [category_column, 'Count']
                val_col = 'Count'
            else:
                if value_column not in df.columns:
                    return create_empty_figure(f"Value column '{value_column}' not found")
                agg_df = df.groupby(category_column)[value_column].agg(aggregation).reset_index()
                agg_df.columns = [category_column, value_column]
                val_col = value_column
            
            agg_df = agg_df.nlargest(top_n, val_col)
            
            if agg_df.empty:
                return create_empty_figure("No data after aggregation")
            
            # Convert to lists
            cat_data = agg_df[category_column].astype(str).tolist()
            val_data = agg_df[val_col].tolist()
            
            fig = go.Figure()
            
            if plot_type == 'pie':
                fig.add_trace(go.Pie(
                    labels=cat_data, values=val_data,
                    hole=0.3, marker=dict(colors=COLOR_PALETTE[:len(cat_data)])
                ))
            else:
                # Sort for horizontal bar
                sorted_idx = np.argsort(val_data)
                cat_sorted = [cat_data[i] for i in sorted_idx]
                val_sorted = [val_data[i] for i in sorted_idx]
                
                fig.add_trace(go.Bar(
                    x=val_sorted, y=cat_sorted,
                    orientation='h', marker_color=COLOR_PALETTE[0]
                ))
                fig.update_layout(xaxis_title=val_col, yaxis_title=category_column)
            
            fig.update_layout(
                title=title or f'{val_col} by {category_column}',
                template='plotly_white', height=max(400, top_n * 35)
            )
            return fig
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def get_required_params(self) -> List[str]:
        return ['category_column']


class ScatterPlotStrategy(VisualizationStrategy):
    """Scatter plot visualization strategy."""
    
    def create(self, df: pd.DataFrame, x_column: str, y_column: str,
               color_column: str = None, title: str = None) -> go.Figure:
        if df is None or df.empty:
            return create_empty_figure("No data available")
        if x_column not in df.columns or y_column not in df.columns:
            return create_empty_figure("Selected columns not found")
        
        try:
            # Prepare data
            plot_df = df[[x_column, y_column]].copy()
            if color_column and color_column in df.columns:
                plot_df[color_column] = df[color_column]
            plot_df = plot_df.dropna(subset=[x_column, y_column])
            
            if len(plot_df) == 0:
                return create_empty_figure("No valid data points")
            
            # Calculate correlation
            corr = float(plot_df[x_column].corr(plot_df[y_column]))
            
            # Convert to lists
            x_data = plot_df[x_column].tolist()
            y_data = plot_df[y_column].tolist()
            
            fig = go.Figure()
            
            if color_column and color_column in plot_df.columns:
                for i, group in enumerate(plot_df[color_column].unique()):
                    mask = plot_df[color_column] == group
                    fig.add_trace(go.Scatter(
                        x=plot_df.loc[mask, x_column].tolist(),
                        y=plot_df.loc[mask, y_column].tolist(),
                        mode='markers', name=str(group),
                        marker=dict(size=8, opacity=0.7, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='markers', name='Data',
                    marker=dict(color=COLOR_PALETTE[0], size=8, opacity=0.7)
                ))
            
            # Trend line
            x_arr = np.array(x_data)
            y_arr = np.array(y_data)
            z = np.polyfit(x_arr, y_arr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
            
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=p(x_line).tolist(),
                mode='lines', name=f'Trend (r={corr:.3f})',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=title or f'{y_column} vs {x_column} (r={corr:.3f})',
                xaxis_title=x_column, yaxis_title=y_column,
                template='plotly_white', height=500
            )
            return fig
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def get_required_params(self) -> List[str]:
        return ['x_column', 'y_column']


class CorrelationHeatmapStrategy(VisualizationStrategy):
    """Correlation heatmap visualization strategy."""
    
    def create(self, df: pd.DataFrame, columns: List[str] = None, title: str = None) -> go.Figure:
        if df is None or df.empty:
            return create_empty_figure("No data available")
        
        try:
            if columns is None:
                columns = get_column_types(df)['numerical']
            
            if len(columns) < 2:
                return create_empty_figure("Need at least 2 numerical columns")
            
            corr = df[columns].corr()
            col_names = corr.columns.tolist()
            z_values = corr.values.tolist()
            
            fig = go.Figure(data=go.Heatmap(
                z=z_values, x=col_names, y=col_names,
                colorscale='RdBu_r', zmin=-1, zmax=1, zmid=0,
                hovertemplate='%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>',
                colorbar=dict(title='Correlation')
            ))
            
            # Add annotations
            for i, row in enumerate(z_values):
                for j, val in enumerate(row):
                    fig.add_annotation(
                        x=col_names[j], y=col_names[i],
                        text=f"{val:.2f}",
                        font=dict(color='white' if abs(val) > 0.5 else 'black', size=10),
                        showarrow=False
                    )
            
            fig.update_layout(
                title=title or 'Correlation Matrix',
                template='plotly_white', height=max(500, len(columns) * 50),
                xaxis=dict(tickangle=45), yaxis=dict(autorange='reversed')
            )
            return fig
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def get_required_params(self) -> List[str]:
        return []


# =============================================================================
# VISUALIZATION FACTORY
# =============================================================================

class VisualizationFactory:
    """Factory to create visualization strategies."""
    
    _strategies = {
        'time_series': TimeSeriesStrategy,
        'distribution': DistributionStrategy,
        'category': CategoryAnalysisStrategy,
        'scatter': ScatterPlotStrategy,
        'correlation': CorrelationHeatmapStrategy
    }
    
    @classmethod
    def get_strategy(cls, chart_type: str) -> VisualizationStrategy:
        strategy_class = cls._strategies.get(chart_type)
        if strategy_class is None:
            raise ValueError(f"Unknown chart type: {chart_type}")
        return strategy_class()
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        return list(cls._strategies.keys())


# =============================================================================
# CHART MANAGER
# =============================================================================

class ChartManager:
    """Manages chart creation and export."""
    
    def __init__(self):
        self.factory = VisualizationFactory()
        self.last_figure: Optional[go.Figure] = None
    
    def create_chart(self, chart_type: str, df: pd.DataFrame, **kwargs) -> go.Figure:
        try:
            strategy = self.factory.get_strategy(chart_type)
            self.last_figure = strategy.create(df, **kwargs)
            return self.last_figure
        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}")
    
    def export_to_png(self, fig: go.Figure = None, filename: str = 'chart.png') -> str:
        if fig is None:
            fig = self.last_figure
        if fig is None:
            raise ValueError("No figure to export")
        fig.write_image(filename, width=1200, height=700, scale=2)
        return filename
    
    def export_to_html(self, fig: go.Figure = None, filename: str = 'chart.html') -> str:
        if fig is None:
            fig = self.last_figure
        if fig is None:
            raise ValueError("No figure to export")
        fig.write_html(filename)
        return filename
    
    def get_available_chart_types(self) -> Dict[str, str]:
        return CHART_TYPES.copy()