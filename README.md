# Inventory Analytics Project - StockSmart

A comprehensive SQL-driven solution for inventory management optimization using advanced analytics and machine learning.

**Live Dashboard**: [View Interactive Dashboard](https://summer-project-2025-ca-iit-guwahati-cjyzxcfoczvcuhjh.streamlit.app)

## Meet Our Team

- **Rahul Jat**
- **Pranav Jori**  
- **Vanshita Bihani** 
- **Arpit Kumar** 

*Team StockSmart - Summer Project 2025, CA IIT Guwahati*

## Project Overview

This project addresses inventory inefficiencies for Urban Retail Co. through advanced SQL analytics, real-time monitoring, and machine learning-based forecasting. The solution transforms raw transactional data into actionable business insights for inventory optimization.

### Key Problem Statement
Urban Retail Co., a growing retail chain, faced inventory management inefficiencies including:
- Frequent stockouts leading to lost sales
- Overstocking causing increased holding costs
- Poor SKU visibility across stores
- Reactive rather than proactive decision-making
- Underutilized sales and logistics data

## Features

### SQL Analytics Engine
- **Daily Inventory Snapshots** - Real-time stock level monitoring
- **Low Inventory Alerts** - Proactive stockout prevention
- **Inventory Turnover Analysis** - Efficiency metrics by product/store
- **Forecast Deviation Tracking** - Demand prediction accuracy assessment
- **Stockout Identification** - Zero inventory incident analysis
- **Overstock Detection** - Excess inventory identification

### Machine Learning Model
- **Algorithm**: XGBoost Classifier for forecast accuracy prediction
- **Features**: demand_forecast, price, discount, holiday_promotion, weather_condition, seasonality
- **Target Classes**: Accurate, Overforecasted, Underforecasted
- **Performance**: 74% test accuracy with minimal overfitting
- **Business Impact**: Enables proactive inventory adjustments

### Interactive Dashboard
- **Real-time KPI Monitoring** - Total inventory, SKU count, turnover metrics
- **Turnover Analysis** - Top/bottom performing SKUs visualization
- **Inventory Trend Tracking** - Time-series analysis with seasonal patterns
- **Forecast Accuracy Insights** - ML-powered prediction classifications
- **Store Comparison** - Cross-location performance analysis

## Project Structure

```
SQL-Inventory-Analytics-Project/
├── dashboard.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── SQL queries/             # Core analytical queries
│   ├── forecast deviation.sql
│   ├── inventory turnover.sql
│   ├── LowInventory.sql
│   ├── Overstockk.sql
│   └── Stockout.sql
├── ML MODEL/                # Machine learning components
│   ├── train.py
│   ├── predict.py
│   ├── forecast_accuracy_model.pkl
│   ├── feature_encoders.pkl
│   └── target_encoder.pkl
├── dashboard/               # Dashboard data and visualizations
│   ├── Low Inventory.csv
│   ├── Overstock.csv
│   ├── Inventory Turnover Rate.csv
│   ├── Forecast Deviation.csv
│   └── graphs/
└── Datasets/               # Raw and processed datasets
    └── inventory_forecasting.csv
```

## Database Schema

The project utilizes a denormalized inventory forecasting dataset with the following key attributes:

**Core Tables:**
- `inventory_forecasting` - Main transactional data
- `Store` - Store location and regional data
- `Product` - Product categorization

**Key Fields:**
- Date, Store_ID, Product_ID
- Inventory_Level, Units_Sold, Units_Ordered
- Demand_Forecast, Price, Discount
- Weather_Condition, Holiday_Promotion, Seasonality
- Competitor_Pricing

## Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rahuljat27/SQL-Inventory-Analytics-Project.git
cd SQL-Inventory-Analytics-Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard:**
```bash
streamlit run dashboard.py
```

4. **Access locally:**
Open your browser and navigate to `http://localhost:8501`

## Key SQL Queries

### 1. Inventory Snapshot
```sql
SELECT date, store_id, product_id, inventory_level, units_sold, units_ordered
FROM inventory_forecasting
ORDER BY date, store_id, product_id;
```

### 2. Low Inventory Alerts
Identifies products where current inventory < 7-day average units sold

### 3. Inventory Turnover Rate
```sql
SELECT store_id, product_id,
       ROUND(CAST(SUM(units_sold) AS FLOAT) / NULLIF(AVG(inventory_level), 0), 2) AS inventory_turnover
FROM inventory_forecasting
WHERE date BETWEEN DATEADD(DAY, -7, '2022-12-24') AND '2022-12-31'
GROUP BY store_id, product_id
ORDER BY inventory_turnover DESC;
```

### 4. Forecast Deviation Analysis
Compares actual units sold with demand forecasts to identify prediction accuracy

### 5. Stockout Detection
Flags all instances where products had zero inventory

### 6. Overstock Identification
Identifies products with high inventory but low movement

## Key Business Insights

### Performance Metrics
- **58,000+ overforecasted instances** vs 82 underforecasted - indicating conservative forecasting bias
- **Store S004** demonstrates best inventory management practices with 4 products in top 10 turnover
- **Products P0069, P0094, P0178** are consistent high performers across multiple stores
- **Seasonal patterns** identified with spikes in Q1, Q4 2022, and Q3 2023

### Strategic Recommendations
1. **Rebalance Forecasting**: Address overforecasting bias to reduce holding costs
2. **Best Practice Replication**: Scale Store S004's strategies across other locations
3. **SKU Optimization**: Focus on high-turnover products for promotional campaigns
4. **Inventory Policies**: Implement customized stocking strategies per SKU rather than uniform approaches

## Technologies Used

- **SQL**: Advanced analytical queries and business intelligence
- **Python**: Data processing, analysis, and machine learning
- **XGBoost**: Gradient boosting for forecast classification
- **Streamlit**: Interactive web dashboard framework
- **Plotly**: Dynamic data visualizations and charts
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning preprocessing and metrics
- **Joblib**: Model serialization and deployment

## Machine Learning Model Details

### Model Architecture
- **Base Algorithm**: XGBoost Classifier
- **Training Data**: 109,500 rows with 87,600 training samples
- **Validation**: Stratified 80/20 train-test split
- **Cross-validation**: Maintained class distribution integrity

### Feature Engineering
- **Numerical Features**: demand_forecast, price, discount (standardized)
- **Categorical Features**: holiday_promotion, weather_condition, seasonality (label encoded)
- **Target Encoding**: 3-class classification (Accurate/Overforecasted/Underforecasted)

### Model Performance
- **Training Accuracy**: 76%
- **Test Accuracy**: 74%
- **Generalization**: Low overfitting with consistent performance
- **Deployment**: Serialized models with joblib for production use

## Dashboard Features

### KPI Dashboard
- Total inventory levels across all locations
- Unique SKU count and diversity metrics
- Average inventory turnover ratios
- Percentage of high-performing SKUs

### Interactive Visualizations
- **Turnover Analysis**: Top 10 and bottom 10 SKU performance
- **Heatmaps**: Store vs Product performance matrices
- **Time Series**: Inventory trends with seasonal analysis
- **Scatter Plots**: Stock vs demand relationship analysis
- **Forecast Accuracy**: ML model prediction distributions

### Filtering Capabilities
- Date range selection for temporal analysis
- Store-specific performance drilling
- Product category filtering
- Theme customization (Light/Dark mode)

## Future Enhancements

### Technical Roadmap
- **Real-time Data Integration**: Live inventory feeds from POS systems
- **Advanced ML Models**: LSTM for time-series forecasting, demand sensing
- **Automated Reordering**: Dynamic safety stock optimization
- **Multi-location Optimization**: Supply chain network analysis

### Business Expansion
- **Supplier Integration**: Vendor performance analytics
- **Customer Segmentation**: Demand pattern analysis by demographics
- **Pricing Optimization**: Dynamic pricing based on inventory levels
- **Promotional Planning**: Campaign impact on inventory flow

## Contributing

We welcome contributions to improve this inventory analytics solution:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Contribution Guidelines
- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings for new functions
- Include unit tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CA IIT Guwahati** - Summer Project Program 2025
- **Urban Retail Co.** - Problem statement and domain expertise
- **Open Source Community** - Libraries and frameworks utilized

## Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [GitHub](https://github.com/rahuljat27/SQL-Inventory-Analytics-Project)
- **Live Dashboard**: [Streamlit App](https://summer-project-2025-ca-iit-guwahati-cjyzxcfoczvcuhjh.streamlit.app)


---

*Built with passion by Team StockSmart | Transforming inventory management through data-driven insights*
