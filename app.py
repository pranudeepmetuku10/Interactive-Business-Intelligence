"""
Business Intelligence Dashboard Application using Gradio
All requirements implemented: Data Upload, Statistics, Filtering, Visualizations, Insights, Export
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tempfile
import os

from data_processor import DataLoader, DataProfiler, DataFilterEngine
from visualizations import ChartManager
from insights import InsightEngine
from utils import get_column_types, format_number, AGGREGATION_METHODS, DEFAULT_PREVIEW_ROWS

# Initialize components
data_loader = DataLoader()
data_profiler = DataProfiler()
filter_engine = DataFilterEngine()
chart_manager = ChartManager()
insight_engine = InsightEngine()

# Global state
current_df = None
filtered_df = None
current_chart = None  # Store current chart for export


# =============================================================================
# DATA UPLOAD FUNCTIONS
# =============================================================================

def load_data(file):
    """Load and preview uploaded data file."""
    global current_df, filtered_df
    
    if file is None:
        empty_update = gr.update(choices=[])
        return ("⚠️ Upload a file", "", "", 
                empty_update, gr.update(value=""), gr.update(value=""),
                empty_update, empty_update,
                empty_update, gr.update(value=""), gr.update(value=""),
                empty_update, empty_update, empty_update,
                empty_update, empty_update,
                empty_update, empty_update, empty_update,
                empty_update, empty_update, empty_update)
    
    try:
        df, msg = data_loader.load_file(file)
        if df is None:
            empty_update = gr.update(choices=[])
            return (f"❌ {msg}", "", "",
                    empty_update, gr.update(value=""), gr.update(value=""),
                    empty_update, empty_update,
                    empty_update, gr.update(value=""), gr.update(value=""),
                    empty_update, empty_update, empty_update,
                    empty_update, empty_update,
                    empty_update, empty_update, empty_update,
                    empty_update, empty_update, empty_update)
        
        current_df = df
        filtered_df = df.copy()
        
        col_types = get_column_types(df)
        num_cols = col_types['numerical']
        cat_cols = col_types['categorical']
        date_cols = col_types['datetime']
        
        # Dataset info
        info = f"""### 📊 Dataset Overview
| Metric | Value |
|--------|-------|
| **Rows** | {len(df):,} |
| **Columns** | {len(df.columns)} |
| **Numerical** | {len(num_cols)} |
| **Categorical** | {len(cat_cols)} |
| **Datetime** | {len(date_cols)} |
| **Missing Values** | {df.isnull().sum().sum():,} |
| **Duplicate Rows** | {df.duplicated().sum():,} |
"""
        
        preview = df.head(DEFAULT_PREVIEW_ROWS).to_markdown(index=False)
        
        # Set default min/max values for numerical filter
        if num_cols:
            first_num = num_cols[0]
            min_val = float(df[first_num].min())
            max_val = float(df[first_num].max())
            num_min_update = gr.update(value=str(min_val))
            num_max_update = gr.update(value=str(max_val))
        else:
            num_min_update = gr.update(value="")
            num_max_update = gr.update(value="")
        
        return (f"✅ {msg}", info, preview,
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None), 
                num_min_update, num_max_update,
                gr.update(choices=cat_cols, value=cat_cols[0] if cat_cols else None),
                gr.update(choices=[]),  # cat_vals - will be populated on cat_col change
                gr.update(choices=date_cols, value=date_cols[0] if date_cols else None),
                gr.update(value=""), gr.update(value=""),  # date start/end
                gr.update(choices=date_cols, value=date_cols[0] if date_cols else None),
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None),
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None),
                gr.update(choices=cat_cols, value=cat_cols[0] if cat_cols else None),
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None),
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None),
                gr.update(choices=num_cols, value=num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)),
                gr.update(choices=cat_cols, value=None),
                gr.update(choices=num_cols, value=num_cols[0] if num_cols else None),
                gr.update(choices=cat_cols, value=cat_cols[0] if cat_cols else None),
                gr.update(choices=date_cols, value=date_cols[0] if date_cols else None))
                
    except Exception as e:
        empty_update = gr.update(choices=[])
        return (f"❌ {str(e)}", "", "",
                empty_update, gr.update(value=""), gr.update(value=""),
                empty_update, empty_update,
                empty_update, gr.update(value=""), gr.update(value=""),
                empty_update, empty_update, empty_update,
                empty_update, empty_update,
                empty_update, empty_update, empty_update,
                empty_update, empty_update, empty_update)


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def generate_stats():
    """Generate comprehensive statistics with missing values report."""
    global current_df
    if current_df is None:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5, showarrow=False)
        return "Upload data first", "Upload data first", "Upload data first", fig
    
    profile = data_profiler.generate_profile(current_df)
    
    # Numerical statistics (with std, quartiles)
    ns = profile.get('numerical_stats', {})
    if ns:
        rows = ["| Column | Mean | Median | Std | Min | Q1 | Q3 | Max | Outliers |",
                "|--------|------|--------|-----|-----|----|----|-----|----------|"]
        for c, s in ns.items():
            rows.append(f"| {c[:12]} | {format_number(s['mean'])} | {format_number(s['median'])} | "
                       f"{format_number(s['std'])} | {format_number(s['min'])} | {format_number(s['q1'])} | "
                       f"{format_number(s['q3'])} | {format_number(s['max'])} | {s['outliers']} |")
        num_md = "### 📈 Numerical Statistics\n\n" + "\n".join(rows)
    else:
        num_md = "### 📈 No numerical columns found"
    
    # Categorical statistics
    cs = profile.get('categorical_stats', {})
    if cs:
        rows = ["| Column | Unique | Mode | Mode Freq |",
                "|--------|--------|------|-----------|"]
        for c, s in cs.items():
            rows.append(f"| {c[:12]} | {s['unique']} | {str(s['mode'])[:15]} | {s['mode_frequency']} |")
        cat_md = "### 📊 Categorical Statistics\n\n" + "\n".join(rows)
    else:
        cat_md = "### 📊 No categorical columns found"
    
    # Missing values report
    missing = profile.get('missing_values', {})
    missing_cols = {k: v for k, v in missing.items() if v['count'] > 0}
    if missing_cols:
        rows = ["| Column | Missing Count | Percentage |",
                "|--------|---------------|------------|"]
        for col, info in sorted(missing_cols.items(), key=lambda x: x[1]['percentage'], reverse=True):
            rows.append(f"| {col[:15]} | {info['count']:,} | {info['percentage']:.1f}% |")
        missing_md = "### ⚠️ Missing Values Report\n\n" + "\n".join(rows)
    else:
        missing_md = "### ✅ No Missing Values\n\nAll columns are complete!"
    
    # Correlation heatmap
    corr = profile.get('correlations')
    if corr is not None and len(corr.columns) >= 2:
        col_names = corr.columns.tolist()
        z_values = corr.values.tolist()
        
        annotations = []
        for i, row in enumerate(z_values):
            for j, val in enumerate(row):
                annotations.append(dict(
                    x=col_names[j], y=col_names[i], text=f"{val:.2f}",
                    font=dict(color='white' if abs(val) > 0.5 else 'black', size=9),
                    showarrow=False
                ))
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values, x=col_names, y=col_names,
            colorscale='RdBu_r', zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title='r')
        ))
        fig.update_layout(
            title='Correlation Matrix', height=500,
            template='plotly_white', annotations=annotations,
            xaxis=dict(tickangle=45), yaxis=dict(autorange='reversed')
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Need 2+ numerical columns", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300)
    
    return num_md, cat_md, missing_md, fig


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def update_num_range(num_col):
    """Update min/max textboxes when numerical column changes."""
    global current_df
    if current_df is None or not num_col or num_col not in current_df.columns:
        return gr.update(value=""), gr.update(value="")
    
    min_val = float(current_df[num_col].min())
    max_val = float(current_df[num_col].max())
    return gr.update(value=str(min_val)), gr.update(value=str(max_val))


def update_cat_values(cat_col):
    """Update categorical values dropdown."""
    global current_df
    if current_df is None or not cat_col or cat_col not in current_df.columns:
        return gr.update(choices=[], value=[])
    
    values = current_df[cat_col].dropna().unique().tolist()
    # Don't pre-select all values - let user choose
    return gr.update(choices=values, value=[])


def update_date_range(date_col):
    """Update date range inputs."""
    global current_df
    if current_df is None or not date_col or date_col not in current_df.columns:
        return gr.update(value=None), gr.update(value=None)
    
    try:
        dates = pd.to_datetime(current_df[date_col])
        min_date = dates.min().strftime('%Y-%m-%d')
        max_date = dates.max().strftime('%Y-%m-%d')
        return gr.update(value=min_date), gr.update(value=max_date)
    except Exception:
        return gr.update(value=""), gr.update(value="")


def apply_filter(num_col, num_min, num_max, cat_col, cat_vals, date_col, date_start, date_end):
    """Apply all filters including date range."""
    global current_df, filtered_df
    
    if current_df is None:
        return "⚠️ No data loaded", ""
    
    filter_engine.clear_filters()
    filters_applied = []
    
    # Numerical filter - parse string inputs.
    if num_col and num_min and num_max:
        try:
            min_val = float(num_min)
            max_val = float(num_max)
            filter_engine.add_filter(num_col, 'numerical', (min_val, max_val))
            filters_applied.append(f"**{num_col}**: {min_val:.2f} - {max_val:.2f}")
        except (ValueError, TypeError):
            pass  # Skip if invalid numbers
    
    # Categorical filter
    if cat_col and cat_vals and len(cat_vals) > 0:
        filter_engine.add_filter(cat_col, 'categorical', cat_vals)
        filters_applied.append(f"**{cat_col}**: {len(cat_vals)} values selected")
    
    # Date filter
    if date_col and (date_start or date_end):
        filter_engine.add_filter(date_col, 'datetime', (date_start, date_end))
        filters_applied.append(f"**{date_col}**: {date_start or 'start'} to {date_end or 'end'}")
    
    filtered_df = filter_engine.apply_filters(current_df)
    
    pct = len(filtered_df) / len(current_df) * 100 if len(current_df) > 0 else 0
    
    status = f"### 🔍 Filter Results\n\n**{len(filtered_df):,}** of {len(current_df):,} rows ({pct:.1f}%)\n\n"
    if filters_applied:
        status += "**Active Filters:**\n" + "\n".join([f"- {f}" for f in filters_applied])
    else:
        status += "*No filters applied*"
    
    preview = filtered_df.head(15).to_markdown(index=False) if len(filtered_df) > 0 else "*No matching rows*"
    
    return status, preview


def clear_filter():
    """Clear all filters."""
    global current_df, filtered_df
    if current_df is None:
        return "No data loaded", ""
    
    filtered_df = current_df.copy()
    filter_engine.clear_filters()
    return f"### ✅ Filters Cleared\n\nShowing all **{len(filtered_df):,}** rows", filtered_df.head(15).to_markdown(index=False)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def make_timeseries(date_col, val_col, agg):
    """Create time series chart."""
    global filtered_df, current_chart
    if filtered_df is None or not date_col or not val_col:
        fig = go.Figure()
        fig.add_annotation(text="Select date and value columns", x=0.5, y=0.5, showarrow=False)
        return fig
    current_chart = chart_manager.create_chart('time_series', filtered_df, 
                                                date_column=date_col, value_column=val_col, aggregation=agg)
    return current_chart


def make_dist(col, ptype, bins):
    """Create distribution chart."""
    global filtered_df, current_chart
    if filtered_df is None or not col:
        fig = go.Figure()
        fig.add_annotation(text="Select a column", x=0.5, y=0.5, showarrow=False)
        return fig
    current_chart = chart_manager.create_chart('distribution', filtered_df, 
                                                column=col, plot_type=ptype.lower(), bins=int(bins))
    return current_chart


def make_category(cat_col, val_col, agg, ctype, topn):
    """Create category chart."""
    global filtered_df, current_chart
    if filtered_df is None or not cat_col:
        fig = go.Figure()
        fig.add_annotation(text="Select a category column", x=0.5, y=0.5, showarrow=False)
        return fig
    current_chart = chart_manager.create_chart('category', filtered_df, 
                                                category_column=cat_col, 
                                                value_column=val_col if val_col else None,
                                                aggregation=agg, plot_type=ctype.lower(), top_n=int(topn))
    return current_chart


def make_scatter(x, y, color):
    """Create scatter plot."""
    global filtered_df, current_chart
    if filtered_df is None or not x or not y:
        fig = go.Figure()
        fig.add_annotation(text="Select X and Y columns", x=0.5, y=0.5, showarrow=False)
        return fig
    current_chart = chart_manager.create_chart('scatter', filtered_df, 
                                                x_column=x, y_column=y, 
                                                color_column=color if color else None)
    return current_chart


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_csv():
    """Export filtered data to CSV."""
    global filtered_df
    if filtered_df is None or filtered_df.empty:
        return None
    
    tmp = tempfile.NamedTemporaryFile(suffix='_filtered_data.csv', delete=False)
    filtered_df.to_csv(tmp.name, index=False)
    return tmp.name


def export_chart_png():
    """Export current chart to PNG."""
    global current_chart
    if current_chart is None:
        return None
    
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='_chart.png', delete=False)
        current_chart.write_image(tmp.name, width=1200, height=700, scale=2)
        return tmp.name
    except Exception as e:
        print(f"PNG export error: {e}")
        return None


# =============================================================================
# INSIGHTS FUNCTIONS
# =============================================================================

def make_insights(val, cat, date):
    """Generate automated insights."""
    global filtered_df
    if filtered_df is None:
        return "⚠️ Upload data first to generate insights."
    
    insights = insight_engine.generate_all_insights(
        filtered_df,
        value_column=val if val else None,
        category_column=cat if cat else None,
        date_column=date if date else None
    )
    return insight_engine.insights_to_markdown(insights)


# =============================================================================
# GRADIO UI
# =============================================================================

with gr.Blocks(title="BI Dashboard", css="""
    /* Modern Dark Professional Theme - Fixed Contrast */
    .gradio-container {
        font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        min-height: 100vh;
    }
    
    /* Main title styling - Centered and Professional */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 50%, #e040fb 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        text-align: center !important;
        letter-spacing: -0.5px !important;
        padding: 20px 0 !important;
        margin-bottom: 10px !important;
    }
    
    /* Section headers - Centered */
    h2 {
        color: #00d4ff !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #7b2cbf !important;
        padding-bottom: 8px !important;
        margin-top: 20px !important;
        text-align: center !important;
    }
    
    h3 {
        color: #e040fb !important;
        font-size: 1.3rem !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    
    /* Tabs styling */
    .tabs {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }
    
    .tab-nav {
        background: transparent !important;
    }
    
    .tab-nav button {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        margin: 4px !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Buttons */
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .gr-button-secondary, .gr-button {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Form elements - Dark inputs with light text */
    .gr-box, .gr-form, .gr-panel {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Input fields - Subtle dark background matching theme */
    input, textarea, select, .gr-input, .gr-text-input {
        background: rgba(15, 20, 40, 0.95) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 8px !important;
        color: #00d4ff !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: rgba(255,255,255,0.4) !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3) !important;
        outline: none !important;
    }
    
    /* Dropdown styling */
    .gr-dropdown, .dropdown-container {
        background: rgba(15, 20, 40, 0.95) !important;
        color: #00d4ff !important;
    }
    
    /* Dropdown options/menu */
    .gr-dropdown ul, .dropdown-menu, [data-testid="dropdown"], .options {
        background: rgba(15, 20, 40, 0.98) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        color: #00d4ff !important;
    }
    
    .gr-dropdown li, .dropdown-item, option {
        color: #e0e0e0 !important;
        background: transparent !important;
    }
    
    .gr-dropdown li:hover, .dropdown-item:hover {
        background: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Multi-select tags/chips - Dark purple with white text */
    .token, .tag, .chip, [data-testid="tag"], .multiselect-tag {
        background: rgba(102, 126, 234, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.9) !important;
        font-weight: 500 !important;
    }
    
    /* Specific fix for Gradio multiselect tokens */
    .token-remove, .remove-tag {
        color: #ffffff !important;
    }
    
    span.token, div.token {
        background: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Labels - Bright and visible */
    label, .gr-label, .label-text {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    /* All text content - Light colors */
    p, span, .prose, .markdown-text, div {
        color: #e0e0e0 !important;
    }
    
    /* Specific markdown prose styling */
    .prose p, .prose span, .prose li {
        color: #e0e0e0 !important;
    }
    
    .prose strong, .prose b {
        color: #ffffff !important;
    }
    
    .prose em, .prose i {
        color: #c0c0c0 !important;
        display: block !important;
        text-align: center !important;
        margin-bottom: 15px !important;
    }
    
    /* Tables in markdown - High contrast and centered */
    table {
        background: rgba(20, 20, 40, 0.8) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border-collapse: collapse !important;
        margin: 0 auto !important;
    }
    
    th {
        background: rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    td {
        color: #e0e0e0 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 10px !important;
        background: rgba(30, 30, 50, 0.5) !important;
    }
    
    tr:hover td {
        background: rgba(102, 126, 234, 0.15) !important;
    }
    
    /* Code blocks */
    code, .code {
        background: rgba(0, 0, 0, 0.3) !important;
        color: #00d4ff !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Plot containers */
    .gr-plot {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    /* File upload area */
    .file-upload, [data-testid="file"] {
        background: rgba(30, 30, 50, 0.6) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        color: #e0e0e0 !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.15) !important;
        margin: 24px 0 !important;
    }
    
    /* Footer hide */
    footer {display: none !important;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    /* Status messages */
    .message, .status {
        color: #e0e0e0 !important;
    }
    
    /* Ensure all nested spans and text are visible */
    * {
        --tw-text-opacity: 1 !important;
    }
    
    .dark span, .dark p, .dark label {
        color: #e0e0e0 !important;
    }
""") as demo:
    gr.Markdown("#  Interactive Business Intelligence Dashboard ")
    gr.Markdown("*Upload CSV or Excel files to explore, analyze, and visualize your data with powerful insights.*")
    
    # =========================================================================
    # TAB 1: UPLOAD
    # =========================================================================
    with gr.Tab("📤 Upload"):
        gr.Markdown("## 📁 Upload Your Data")
        gr.Markdown("*Supported formats: CSV, Excel (.xlsx, .xls). Maximum recommended file size: 50MB.*")
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload CSV/Excel", file_types=[".csv", ".xlsx", ".xls"])
            with gr.Column(scale=2):
                status = gr.Markdown("*Waiting for file upload...*")
        info = gr.Markdown()
        gr.Markdown("## 👀 Data Preview")
        gr.Markdown("*First 10 rows of your uploaded dataset.*")
        preview = gr.Markdown()
    
    # =========================================================================
    # TAB 2: STATISTICS
    # =========================================================================
    with gr.Tab("📈 Statistics"):
        gr.Markdown("## 📊 Statistical Analysis")
        gr.Markdown("*Comprehensive statistics for all columns in your dataset.*")
        stat_btn = gr.Button("🔄 Generate Statistics", variant="primary")
        with gr.Row():
            with gr.Column():
                num_out = gr.Markdown()
            with gr.Column():
                cat_out = gr.Markdown()
        missing_out = gr.Markdown()
        gr.Markdown("## 🔗 Correlation Matrix")
        gr.Markdown("*Discover relationships between numerical variables. Values range from -1 (negative correlation) to +1 (positive correlation).*")
        corr_plot = gr.Plot()
        
        stat_btn.click(generate_stats, outputs=[num_out, cat_out, missing_out, corr_plot])
    
    # =========================================================================
    # TAB 3: FILTER
    # =========================================================================
    with gr.Tab("🔍 Filter"):
        gr.Markdown("## 🎯 Filter Your Data")
        gr.Markdown("*Narrow down your dataset using multiple filter criteria. Changes apply to visualizations and insights.*")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔢 Numerical Filter")
                num_col = gr.Dropdown(label="Column", choices=[])
                with gr.Row():
                    num_min = gr.Textbox(label="Min Value", value="")
                    num_max = gr.Textbox(label="Max Value", value="")
            with gr.Column():
                gr.Markdown("### 🏷️ Categorical Filter")
                cat_col = gr.Dropdown(label="Column", choices=[])
                cat_vals = gr.Dropdown(label="Values", choices=[], multiselect=True)
            with gr.Column():
                gr.Markdown("### 📅 Date Filter")
                date_col = gr.Dropdown(label="Column", choices=[])
                date_start = gr.Textbox(label="Start Date (YYYY-MM-DD)")
                date_end = gr.Textbox(label="End Date (YYYY-MM-DD)")
        
        with gr.Row():
            apply_btn = gr.Button("🔍 Apply Filters", variant="primary")
            clear_btn = gr.Button("🗑️ Clear All")
        
        filter_status = gr.Markdown()
        filter_preview = gr.Markdown()
        
        gr.Markdown("## 📥 Export Filtered Data")
        gr.Markdown("*Download your filtered dataset as a CSV file.*")
        export_csv_btn = gr.Button("Download Filtered Data (CSV)")
        export_csv_file = gr.File(label="CSV Download")
        
        # Filter events
        num_col.change(update_num_range, inputs=[num_col], outputs=[num_min, num_max])
        cat_col.change(update_cat_values, inputs=[cat_col], outputs=[cat_vals])
        date_col.change(update_date_range, inputs=[date_col], outputs=[date_start, date_end])
        apply_btn.click(apply_filter, inputs=[num_col, num_min, num_max, cat_col, cat_vals, date_col, date_start, date_end], 
                       outputs=[filter_status, filter_preview])
        clear_btn.click(clear_filter, outputs=[filter_status, filter_preview])
        export_csv_btn.click(export_csv, outputs=[export_csv_file])
    
    # =========================================================================
    # TAB 4: VISUALIZATIONS
    # =========================================================================
    with gr.Tab("📊 Visualizations"):
        gr.Markdown("## 📈 Time Series Analysis")
        gr.Markdown("*Visualize trends and patterns in your data over time. Perfect for tracking metrics like sales, revenue, or any value that changes over time.*")
        with gr.Row():
            ts_date = gr.Dropdown(label="Date Column", choices=[])
            ts_val = gr.Dropdown(label="Value Column", choices=[])
            ts_agg = gr.Dropdown(label="Aggregation", choices=AGGREGATION_METHODS, value="sum")
        ts_btn = gr.Button("Create Time Series", variant="primary")
        ts_plot = gr.Plot()
        ts_btn.click(make_timeseries, inputs=[ts_date, ts_val, ts_agg], outputs=[ts_plot])
        
        gr.Markdown("---")
        gr.Markdown("## 📊 Distribution Analysis")
        gr.Markdown("*Understand how your numerical data is spread out. Identify patterns, outliers, and the central tendency of your values.*")
        with gr.Row():
            dist_col = gr.Dropdown(label="Column", choices=[])
            dist_type = gr.Dropdown(label="Type", choices=["histogram", "box"], value="histogram")
            dist_bins = gr.Slider(10, 100, 30, step=5, label="Bins")
        dist_btn = gr.Button("Create Distribution", variant="primary")
        dist_plot = gr.Plot()
        dist_btn.click(make_dist, inputs=[dist_col, dist_type, dist_bins], outputs=[dist_plot])
        
        gr.Markdown("---")
        gr.Markdown("## 📋 Category Analysis")
        gr.Markdown("*Compare performance across different categories. See which groups contribute most to your totals or averages.*")
        with gr.Row():
            cc_col = gr.Dropdown(label="Category Column", choices=[])
            cc_val = gr.Dropdown(label="Value Column (optional)", choices=[])
            cc_agg = gr.Dropdown(label="Aggregation", choices=AGGREGATION_METHODS, value="sum")
        with gr.Row():
            cc_type = gr.Dropdown(label="Chart Type", choices=["bar", "pie"], value="bar")
            cc_top = gr.Slider(5, 25, 10, step=1, label="Top N Categories")
        cc_btn = gr.Button("Create Category Chart", variant="primary")
        cc_plot = gr.Plot()
        cc_btn.click(make_category, inputs=[cc_col, cc_val, cc_agg, cc_type, cc_top], outputs=[cc_plot])
        
        gr.Markdown("---")
        gr.Markdown("## 🔵 Scatter Plot & Correlation")
        gr.Markdown("*Explore relationships between two numerical variables. Discover correlations and patterns in your data.*")
        with gr.Row():
            sc_x = gr.Dropdown(label="X Axis", choices=[])
            sc_y = gr.Dropdown(label="Y Axis", choices=[])
            sc_c = gr.Dropdown(label="Color By (optional)", choices=[])
        sc_btn = gr.Button("Create Scatter Plot", variant="primary")
        sc_plot = gr.Plot()
        sc_btn.click(make_scatter, inputs=[sc_x, sc_y, sc_c], outputs=[sc_plot])
        
        gr.Markdown("---")
        gr.Markdown("## 📥 Export Chart")
        gr.Markdown("*Save your current visualization as a high-resolution PNG image. The last chart you created will be exported.*")
        export_png_btn = gr.Button("Download Current Chart (PNG)")
        export_png_file = gr.File(label="PNG Download")
        export_png_btn.click(export_chart_png, outputs=[export_png_file])
    
    # =========================================================================
    # TAB 5: INSIGHTS
    # =========================================================================
    with gr.Tab("💡 Insights"):
        gr.Markdown("## 🔮 Automated Insights Generation")
        gr.Markdown("*Let AI analyze your data and discover key patterns, trends, and anomalies automatically.*")
        gr.Markdown("### ⚙️ Configure Analysis")
        gr.Markdown("*Select the columns you want to analyze for deeper insights:*")
        with gr.Row():
            ins_val = gr.Dropdown(label="Value Column (for top/bottom analysis)", choices=[])
            ins_cat = gr.Dropdown(label="Category Column (for grouping)", choices=[])
            ins_date = gr.Dropdown(label="Date Column (for trends)", choices=[])
        ins_btn = gr.Button("🔮 Generate Insights", variant="primary", size="lg")
        ins_out = gr.Markdown()
        ins_btn.click(make_insights, inputs=[ins_val, ins_cat, ins_date], outputs=[ins_out])
    
    # =========================================================================
    # FILE UPLOAD EVENT - Updates ALL components
    # =========================================================================
    file_in.change(
        load_data,
        inputs=[file_in],
        outputs=[
            status, info, preview,
            num_col, num_min, num_max, cat_col, cat_vals, date_col, date_start, date_end,
            ts_date, ts_val, dist_col, cc_col, cc_val, sc_x, sc_y, sc_c,
            ins_val, ins_cat, ins_date
        ]
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch()