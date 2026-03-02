# Interactive Business Intelligence Dashboard

A professional, interactive Business Intelligence dashboard built with Python and Gradio that enables non-technical stakeholders to explore and analyze business data through an intuitive web interface.

## Project Overview

This application allows users to:
- Upload CSV and Excel datasets
- View comprehensive data profiling and statistics
- Apply interactive filters (numerical, categorical, date)
- Generate 5 different visualization types
- Extract automated business insights
- Export filtered data and charts

## Target Users

- Business analysts who need quick data exploration
- Non-technical stakeholders requiring visual insights
- Data teams needing rapid prototyping of dashboards
- Anyone who wants to analyze data without coding

## Features

### 1. Data Upload and Validation
- Supports CSV and Excel (.xlsx, .xls) formats
- Automatic data type detection
- Dataset overview with row/column counts
- Missing value and duplicate detection
- Data preview functionality

### 2. Data Profiling and Statistics
- Numerical columns: Mean, median, std, min, max, Q1, Q3, outlier count
- Categorical columns: Unique values, mode, mode frequency
- Missing values report: Count and percentage by column
- Correlation matrix: Interactive heatmap with annotations

### 3. Interactive Filtering
- Numerical filters: Min/Max value inputs
- Categorical filters: Multi-select dropdown
- Date filters: Start/End date range
- Real-time row count updates
- Filter combination support

### 4. Visualizations (5 Types)
1. Time Series Analysis: Trends over time with aggregation options
2. Distribution Analysis: Histogram and box plots
3. Category Analysis: Bar charts and pie charts
4. Scatter Plot: Relationship analysis with optional color coding
5. Correlation Heatmap: Variable relationship matrix

All visualizations include:
- User-selectable columns
- Multiple aggregation methods (sum, mean, median, count, min, max, std)
- Clear titles, labels, and legends
- Interactive Plotly charts

### 5. Automated Insights
- Top/Bottom performers analysis
- Trend analysis with direction detection
- Outlier/Anomaly detection using IQR method
- Strong correlation identification
- Data quality assessment

### 6. Export Functionality
- Export filtered data as CSV
- Export visualizations as PNG images

## Project Structure

```
FINAL_PROJECT/
├── app.py                  # Main Gradio application
├── data_processor.py       # Data loading, cleaning, filtering (Strategy Pattern)
├── visualizations.py       # Chart creation functions (Strategy Pattern)
├── insights.py             # Automated insight generation (Strategy Pattern)
├── utils.py                # Helper functions and constants
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/                   # Sample datasets
    ├── sample_financial_data.csv
    ├── sample_sales_performance.csv
    └── Online Retail.xlsx
```

## Design Pattern: Strategy Pattern

This project implements the Strategy Pattern in three key areas:

### 1. Filter Strategies (data_processor.py)
- `FilterStrategy` - Abstract base class
- `NumericalFilterStrategy` - Range-based filtering
- `CategoricalFilterStrategy` - Multi-select filtering
- `DateTimeFilterStrategy` - Date range filtering
- `BooleanFilterStrategy` - True/False filtering

### 2. Visualization Strategies (visualizations.py)
- `VisualizationStrategy` - Abstract base class
- `TimeSeriesStrategy` - Time-based trend charts
- `DistributionStrategy` - Histogram and box plots
- `CategoryAnalysisStrategy` - Bar and pie charts
- `ScatterPlotStrategy` - Scatter plots with correlation
- `CorrelationHeatmapStrategy` - Correlation matrices

### 3. Insight Strategies (insights.py)
- `InsightStrategy` - Abstract base class
- `TopBottomPerformersStrategy` - Ranking analysis
- `TrendAnalysisStrategy` - Trend detection
- `AnomalyDetectionStrategy` - Outlier identification
- `CorrelationInsightStrategy` - Relationship analysis
- `DataQualityStrategy` - Data quality assessment

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or extract the project:
   ```bash
   cd FINAL_PROJECT
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your virtual environment is activated

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

## Dependencies

| Package    | Version  | Purpose                          |
|------------|----------|----------------------------------|
| pandas     | >=2.0.0  | Data manipulation and analysis   |
| numpy      | >=1.24.0 | Numerical operations             |
| gradio     | ==3.50.2 | Web interface                    |
| plotly     | >=5.15.0 | Interactive visualizations       |
| matplotlib | >=3.7.0  | Static plots                     |
| seaborn    | >=0.12.0 | Statistical visualizations       |
| openpyxl   | >=3.1.0  | Excel file support               |
| xlrd       | >=2.0.0  | Legacy Excel support             |
| scipy      | >=1.10.0 | Statistical functions            |
| kaleido    | >=0.2.1  | Chart export to PNG              |
| tabulate   | >=0.9.0  | Table formatting                 |

## Sample Datasets

### 1. Financial Data (sample_financial_data.csv)
- 1,825 records of stock market data
- Columns: date, symbol, open_price, high_price, low_price, close_price, volume, daily_change, sector, market_cap, pe_ratio, etc.
- Use case: Financial trend analysis and stock comparison

### 2. Sales Performance (sample_sales_performance.csv)
- 1,000 records of sales transactions
- Columns: sale_id, sale_date, sales_rep, region, product, deal_value, quantity, discount_percent, etc.
- Use case: Sales analytics and rep performance tracking

### 3. Online Retail (Online Retail.xlsx)
- E-commerce transaction data
- Use case: Retail analytics and customer behavior analysis

## Usage Guide

### Step 1: Upload Data
- Navigate to the "Upload" tab
- Click "Upload CSV/Excel" and select your file
- Review the dataset overview and preview

### Step 2: View Statistics
- Go to the "Statistics" tab
- Click "Generate Statistics"
- Review numerical stats, categorical stats, and correlation matrix

### Step 3: Filter Data
- Navigate to the "Filter" tab
- Set numerical ranges, select categories, or define date ranges
- Click "Apply Filters" to filter the data
- View the filtered row count and preview

### Step 4: Create Visualizations
- Go to the "Visualizations" tab
- Select columns and chart options
- Click the create button for each visualization type
- Export charts as PNG using the export button

### Step 5: Generate Insights
- Navigate to the "Insights" tab
- Select value, category, and date columns for analysis
- Click "Generate Insights"
- Review automated findings grouped by priority

### Step 6: Export Data
- Use "Download Filtered Data (CSV)" in the Filter tab
- Use "Download Current Chart (PNG)" in the Visualizations tab

## Error Handling

The application handles common issues gracefully:
- Invalid file formats: Clear error message displayed
- Missing values: Reported in statistics, handled in calculations
- Empty filter results: Informative message shown
- Invalid column selections: Appropriate feedback provided


