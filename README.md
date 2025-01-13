# Champagne Insights: Unveiling Sales Trends for Perrin Freres

## **Executive Summary**

At **Perrin Freres**, a prestigious champagne manufacturer, understanding sales trends is critical to maintaining market leadership. As the company's **Data Scientist**, I analyzed historical monthly sales data to uncover seasonal trends and peak demand periods. Leveraging advanced data analysis and forecasting techniques, I identified actionable insights to optimize production schedules and enhance marketing strategies, aligning them with demand patterns.

Key findings from the analysis include:
- **Seasonal Patterns**: High sales volumes in December, driven by holiday celebrations, with noticeable dips in demand during the summer months.  
- **Growth Trajectory**: A consistent upward trend in champagne sales from the mid-1960s to the early 1970s, despite short-term fluctuations.  
- **Strategic Recommendations**: Tailored marketing campaigns and production strategies to capitalize on peak demand periods and address seasonal dips effectively.

By integrating these insights into Perrin Freres' operations, the company can better meet customer expectations during key celebratory moments, ensure operational efficiency, and maintain a competitive edge in the market. This project not only strengthens Perrin Freres’ ability to navigate seasonal demand variations but also lays the groundwork for data-driven decision-making to secure long-term growth and success.

---

## **Objectives**
1. **Trend Analysis**: Identify historical trends in monthly champagne sales to understand overall patterns.
2. **Seasonality Detection**: Analyze recurring seasonal variations to optimize marketing efforts.
3. **Peak Sales Identification**: Highlight high-demand months and investigate contributing factors.
4. **Production Optimization**: Provide recommendations to align production with demand.
5. **Marketing Strategy Development**: Design targeted campaigns based on seasonal insights.
6. **Forecasting Future Sales**: Build a predictive model to estimate future demand patterns.

---

## **Data Collection**

### **Dataset Overview**
The dataset used in this analysis contains monthly sales data for champagne products, capturing the historical performance of **Perrin Freres** over a specified period. It serves as a critical resource for identifying demand patterns, analyzing trends, and supporting strategic business planning.

### **Attributes**
The dataset includes the following columns:
- **Month**: Represents the date of each observation, indicating the year and month of recorded sales.
- **Sales**: Monthly sales figures (units or revenue), serving as the primary metric for trend analysis and forecasting.

### **Data Source**
The dataset was sourced from **Kaggle**, a reputable platform for datasets and machine learning competitions. It reflects historical sales data for Perrin Freres and is assumed to be accurate and reliable for analysis. The data was selected due to its alignment with the project objectives and its completeness for conducting time series analysis.

---

## **Tools and Technologies Used**

This project leverages several tools, technologies, and Python libraries to analyze and model Perrin Freres' monthly champagne sales data. The following tools and techniques were utilized:

### **1. Programming Language**
- **Python**: The primary programming language used for data analysis, visualization, and modeling due to its extensive library support and ease of use.

### **2. Python Libraries**
- **Pandas**: Used for data manipulation, cleaning, and time series analysis. Pandas provided efficient data structures like DataFrames to handle the dataset and perform preprocessing tasks such as parsing dates and extracting features like year and month.
  
- **NumPy**: Utilized for numerical operations and data transformations, ensuring smooth handling of mathematical computations within the project.

- **Matplotlib**: Used for creating static, interactive, and visually appealing plots to analyze trends and patterns in the data.

- **Seaborn**: A statistical data visualization library built on Matplotlib. It was used for creating advanced visualizations, such as box plots and line plots, to better understand seasonal patterns and trends in sales data.

### **3. Time Series Modeling**
- **ARIMA (AutoRegressive Integrated Moving Average)**: Applied to capture and model the time-dependent structure of the data. ARIMA is effective for analyzing and forecasting non-seasonal time series data by combining autoregressive and moving average components.

- **SARIMAX (Seasonal AutoRegressive Integrated Moving-Average with eXogenous regressors)**: Used to account for the seasonality present in the champagne sales data. SARIMAX incorporates seasonal trends and external factors, making it an ideal model for forecasting in this project.

### **4. Development Environment**
- **Jupyter Notebook**: The development and analysis were conducted in Jupyter Notebook, an interactive environment that allows for iterative exploration, visualization, and documentation.

### **5. Visualization and Reporting**
- The visualizations created using Matplotlib and Seaborn include:
  - Monthly sales trends over time
  - Yearly sales trends
  - Monthly sales distribution using box plots
- These plots were instrumental in uncovering patterns and seasonality in the data, which guided the modeling and forecasting processes.

---

## **Exploratory Data Analysis**
The dataset comprises monthly sales data from Perrin Freres, spanning multiple years. Below are key findings from the exploratory analysis:

### **1. Monthly Sales Over Time**
![Monthly Sales Overtime](https://github.com/user-attachments/assets/7fbfc70a-7d70-432f-b05d-62fb0b73002e)


The line plot illustrates monthly champagne sales, revealing distinct seasonal fluctuations. Peaks occur at regular intervals, indicating a recurring pattern in consumer behavior. Overall, the trend highlights periods of growth and decline, reflecting the dynamic nature of the market.

---

### **2. Monthly Sales Distribution**
![Monthly Sales Distribution](https://github.com/user-attachments/assets/6ed58822-c142-44e3-b292-ae865c4216fb)


The box plot visualizes the spread of monthly sales across the year:
- **December** exhibits the highest sales distribution, driven by holiday demand.
- **Summer months** show a dip in sales, indicating a seasonal decline.
These patterns provide valuable insights into demand cycles, enabling strategic planning.

---

### **3. Yearly Sales Trend**
![Yearly Sales Trend](https://github.com/user-attachments/assets/da514bce-e4c1-475b-9c27-39c38ee5cbf6)


The year-wise analysis highlights a steady growth trajectory in champagne sales from the mid-1960s to the early 1970s, with some fluctuations:
- **Peaks** suggest successful marketing campaigns or external factors driving demand.
- **Troughs** may reflect market challenges or off-peak periods.

---

## **Time Series Modeling**

In this section, we applied advanced time series analysis techniques to model and forecast Perrin Freres' monthly champagne sales. The objective was to identify the best model for predicting future sales trends and patterns. Below are the steps undertaken for time series modeling:

---

### **1. Stationarity Check and Data Transformation**

#### **Stationarity Check**
Time series data must be stationary (i.e., statistical properties like mean, variance, and autocovariance remain constant over time) for effective modeling with ARIMA. To check stationarity, the **Augmented Dickey-Fuller (ADF) Test** was used.

```python
# ADF Test Function
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ["ADF Test Statistics", "p-value", "#lags used"]
    for value, label in zip(result, labels):
        print(label + " : " + str(value))
    if result[1] <= 0.05:
        print("Data is stationary and ready for forecasting")
    else:
        print("Data is not stationary and needs transformation")

# Initial ADF Test
adfuller_test(df["Sales"])
```

**Result**:
```
ADF Test Statistics: -1.8335930563276215
p-value: 0.36391577166024586
#lags used: 11
Data is not stationary and needs transformation
```

#### **Data Transformation**
To stabilize the mean and variance of the series, a combination of logarithmic scaling, square root, cube root, and differencing transformations was applied. After these transformations, the data became stationary, as confirmed by the ADF test:

```python
# Apply combined transformations and differencing
df["log_sqrt_cbrt_sales"] = np.log(np.sqrt(np.cbrt(df["Sales"])))
df["shift_log_sqrt_cbrt_sales"] = df["log_sqrt_cbrt_sales"].shift()
df["shift_diff"] = df["log_sqrt_cbrt_sales"] - df["shift_log_sqrt_cbrt_sales"]

# ADF Test after transformation
adfuller_test(df["shift_diff"].dropna())
```

**Result**:
```
ADF Test Statistics: -4.460914465253624
p-value: 0.000231214046495368
#lags used: 12
Data is stationary and ready for forecasting
```

The plot below shows the rolling mean and standard deviation of the transformed data, confirming stationarity:

![Sales Data With Rolling Mean and Standard Deviation](https://github.com/user-attachments/assets/ff8ae841-71d7-4fe4-ad4d-57d6fa28f080)


### **2. Finding Optimal ARIMA Parameters**

To identify the best parameters for the ARIMA model, we used the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots. These helped determine the order of the ARIMA model, represented as (p, d, q).

```python
# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF Plot
plot_acf(df["shift_diff"].dropna(), lags=20)
plt.show()

# PACF Plot
plot_pacf(df["shift_diff"].dropna(), lags=20)
plt.show()
```

![ACF Plot](https://github.com/user-attachments/assets/a6dc21d0-4099-4d44-9112-72e39b8acb4a)

![PACF Plot](https://github.com/user-attachments/assets/91ac509a-01ce-4adb-8e55-534951f95424)


**Optimal Parameters**:
- **p**: 2 (from PACF plot)
- **d**: 1 (from differencing)
- **q**: 4 (from ACF plot)


### **3. ARIMA Model**

#### **Fitting the ARIMA Model**
Using the identified parameters (p=2, d=1, q=4), we fitted the ARIMA model to the data:

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
arima_model = ARIMA(df["log_sqrt_cbrt_sales"].dropna(), order=(2, 1, 4))
arima_result = arima_model.fit()
print(arima_result.summary())
```

However, the ARIMA model did not yield satisfactory forecasting results. Below is a placeholder for the ARIMA forecast visualization:

![ARIMA Forecast vs Actual Forecast](https://github.com/user-attachments/assets/ac730917-4bac-4a34-b2ba-8de37bca84cb)


### **4. SARIMAX Model**

#### **Why SARIMAX?**
Since the data exhibited strong seasonality, we used the **SARIMAX** (Seasonal AutoRegressive Integrated Moving-Average with eXogenous regressors) model, which accounts for both trend and seasonality.

#### **Fitting the SARIMAX Model**
The SARIMAX model was fitted to the data, and predictions were made for the last few months of the available sales period to validate the model's accuracy:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMAX model
sarimax_model = SARIMAX(df["log_sqrt_cbrt_sales"].dropna(), order=(2, 1, 4), seasonal_order=(1, 1, 1, 12))
sarimax_result = sarimax_model.fit()
print(sarimax_result.summary())
```

#### **Results for Last Months**
The SARIMAX model produced accurate predictions for the final months of the dataset, as shown below:

![SARIMAX Forcast vs Actual Forecast](https://github.com/user-attachments/assets/51896891-f6b3-4c62-92cc-61ce118a387e)


This strong performance indicated that the model could be reliably used for future forecasting.

### **5. Forecasting Future Sales**

After validating the model, we forecasted champagne sales for future periods. The SARIMAX model provided excellent results, as demonstrated below:

![Sales Forcast for the Next Financial Period](https://github.com/user-attachments/assets/43e38abe-0140-4e8e-a76c-445d3e74410e)


---


## **Forecasting Results and Insights**

This section presents the outcomes of the SARIMAX model used for forecasting monthly champagne sales for Perrin Freres. The results demonstrate the model's ability to capture the seasonal and trend patterns in the data effectively.

---

### **Forecast Visualization**

The SARIMAX model provided a forecast for future sales, showcasing its performance in accurately predicting patterns over time. Below is the visualization of the SARIMAX forecast:

![Sales Forcast for the Next Financial Period](https://github.com/user-attachments/assets/96d7f5af-e8b1-4395-b835-0b2196ee71e3)


---

### **Forecasted Sales Values**

The following are the forecasted sales values for the months beyond the original dataset period. These predictions will help Perrin Freres plan their production, inventory, and marketing strategies effectively:

| **Month**        | **Forecasted Sales** |
|-------------------|-----------------------|
| 1972-09-01       | 5,992.83             |
| 1972-10-01       | 7,117.50             |
| 1972-11-01       | 10,081.82            |
| 1972-12-01       | 12,937.54            |
| 1973-01-01       | 3,989.42             |
| 1973-02-01       | 3,777.11             |
| 1973-03-01       | 4,668.60             |
| 1973-04-01       | 4,761.86             |
| 1973-05-01       | 4,962.66             |
| 1973-06-01       | 5,288.86             |
| 1973-07-01       | 4,480.23             |
| 1973-08-01       | 1,770.29             |
| 1973-09-01       | 6,130.80             |
| 1973-10-01       | 7,487.40             |
| 1973-11-01       | 10,533.08            |
| 1973-12-01       | 13,477.72            |
| 1974-01-01       | 3,703.91             |

These forecasted values align with the seasonal and cyclic nature of champagne sales, as observed in the historical data. 

---

### **Insights from the Forecast**

- The SARIMAX model captured the high sales periods during the holiday months (e.g., November and December), showcasing peaks in demand.
- Predicted values for non-holiday months (e.g., January and February) reflect lower sales, consistent with historical trends.
- The ability to accurately predict these values provides Perrin Freres with actionable insights for better resource allocation and strategic planning.

---


## **Recommendations and Strategic Actions**

The sales forecast for Perrin Freres champagne provides significant insights into seasonal trends and future demand patterns. Based on the findings of this project, the following recommendations and strategic actions are proposed to address the project objectives and ensure effective inventory management and sales optimization:

---

### **1. Inventory Optimization**

- **Objective Addressed:** Optimize inventory levels to prevent overstocking or stockouts, particularly for seasonal products.  
- **Recommendation:**  
  - **Peak Sales Periods (November–December):** Ensure higher inventory levels are maintained during these months to meet the increased demand predicted by the SARIMAX model. 
  - **Low Sales Periods (January–February):** Reduce inventory to prevent overstocking and minimize holding costs during the non-peak months.  
  - **Action Plan:**  
    - Utilize a rolling forecast system to update inventory needs dynamically as new data becomes available.  
    - Allocate storage and resources for peak season months well in advance.  

---

### **2. Production Planning**

- **Objective Addressed:** Align production schedules with forecasted sales to avoid bottlenecks or surplus production.  
- **Recommendation:**  
  - Scale production during high-demand months, with a lead time of at least two months to allow for distribution.  
  - Focus on smaller production runs for the low-demand months while retaining flexibility for unexpected changes in demand.  
  - Collaborate closely with suppliers to ensure raw materials are available when needed.  
 
---

### **3. Marketing and Sales Strategy**

- **Objective Addressed:** Leverage sales patterns to create effective marketing campaigns and increase overall revenue.  
- **Recommendation:**  
  - **Seasonal Promotions:** Launch targeted marketing campaigns during high-demand months (e.g., November and December) to capitalize on holiday sales trends.  
  - **Off-Season Discounts:** Offer special discounts or bundled offers during low-demand periods to boost sales and reduce stagnant inventory.  
  - **Data-Driven Advertising:** Use historical sales data and forecasts to plan region-specific marketing strategies, focusing on areas with the highest sales potential.  
  - **Digital Campaigns:** Invest in online marketing platforms to promote products effectively and reach a wider audience during key sales months.  

---

### **4. Sales Forecasting and Decision-Making**

- **Objective Addressed:** Provide actionable insights to support data-driven decision-making.  
- **Recommendation:**  
  - Use the SARIMAX model’s forecasting capabilities as an ongoing tool to monitor sales trends.  
  - Regularly update the model with fresh sales data to refine predictions.  
  - Automate forecast reporting to provide stakeholders with actionable insights on a monthly basis.  

---

### **5. Long-Term Strategic Recommendations**

- **Expand Forecasting Scope:**  
  Incorporate external factors such as economic indicators, competitor performance, and weather data into the forecasting process for greater accuracy.  
- **Seasonal Inventory Trends:**  
  Implement a more robust supply chain strategy to account for the variability in peak and low seasons, ensuring efficient logistics and distribution.  
- **Product Diversification:**  
  Explore opportunities to introduce complementary products to reduce reliance on seasonal champagne sales and maintain steady revenue throughout the year.

---

### **Conclusion and Future Actions**

By addressing the project objectives, these recommendations provide a comprehensive approach to improving Perrin Freres' operational efficiency, reducing costs, and increasing revenue. The insights gained from the forecast not only help optimize inventory and production but also enable the business to make data-driven strategic decisions.

Moving forward, Perrin Freres can enhance its forecasting and analytics capabilities by:
- Incorporating machine learning models for improved accuracy.
- Using customer segmentation and purchasing behavior data to refine marketing strategies.
- Regularly evaluating the model’s performance and updating it with the latest data trends.

The adoption of these recommendations ensures that Perrin Freres remains competitive in the market, adapts effectively to seasonal trends, and maximizes profitability.

---


## **Limitations of Work**

While this project provides valuable insights and actionable recommendations for Perrin Freres, there are certain limitations to consider:

1. **Data Coverage and Quality**:  
   - The dataset spans a specific historical period, which may not fully capture recent shifts in consumer behavior or external factors (e.g., economic changes, competitor activities, or global events).  
   - The data is aggregated on a monthly basis, potentially obscuring granular patterns such as weekly or daily fluctuations.

2. **Model Assumptions**:  
   - The ARIMA and SARIMAX models assume linear relationships and stationarity. While transformations were applied to meet these assumptions, non-linear dynamics and sudden changes in trends (e.g., due to external disruptions) may not be fully accounted for.  

3. **Exclusion of External Variables**:  
   - The analysis relies solely on historical sales data and does not incorporate external factors like marketing campaigns, pricing strategies, weather conditions, or consumer sentiment, which could significantly influence demand.  
   - Incorporating such variables in future models could enhance the accuracy of forecasts and provide deeper insights.

4. **Seasonality and Special Events**:  
   - The SARIMAX model effectively captures seasonal patterns, but it may not predict unexpected spikes or dips in sales due to one-time events (e.g., new product launches, regulatory changes, or cultural shifts).  

5. **Generalizability**:  
   - The insights and forecasts generated are specific to Perrin Freres' historical sales data and may not apply to other markets, regions, or industries without further customization.

6. **Limited Forecast Horizon**:  
   - The forecasts generated extend to a limited period, and the accuracy of predictions decreases as the forecast horizon expands. Regular updates to the model with new data will be necessary to maintain reliability over time.

### **Future Considerations**  
To address these limitations, future efforts could focus on:  
- Incorporating external variables and richer datasets to account for broader influences on sales trends.  
- Exploring advanced machine learning models, such as LSTMs or Prophet, to capture complex patterns and non-linear relationships.  
- Conducting additional analyses to evaluate the impact of specific marketing or operational strategies on sales.  
- Updating the dataset regularly to ensure forecasts reflect current market conditions.   

