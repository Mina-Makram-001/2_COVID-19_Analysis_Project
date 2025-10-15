# Project Overview

This project analyzes the **COVID-19 pandemic trends in the United States**, focusing on understanding the relationship between **confirmed, recovered, and death cases** over time.
Using Python’s data analysis and visualization libraries, the project explores how the virus spread, identifies statistical patterns, and tests the strength of relationships between key variables.


## Objectives

* Clean and preprocess real-world COVID-19 data for accurate analysis.
* Visualize trends of confirmed, recovered, and death cases over time.
* Calculate daily new cases and growth rates to capture short-term changes.
* Perform **statistical tests** and **correlation analysis** to understand relationships between variables.
* Apply **linear regression** using `statsmodels` to measure how confirmed cases predict deaths.


## Key Analyses

* **Trend Visualization:** Line plots showing the progression of confirmed, recovered, and death cases in the U.S.
* **Correlation Analysis:** Pearson correlation test between confirmed and death cases
  → *Result: r ≈ 0.99, p ≈ 0*, indicating a strong and statistically significant positive correlation.
* **Regression Modeling:** Built a simple linear regression model to predict deaths from confirmed cases using `statsmodels.OLS`.
* **Hypothesis Testing:** Applied statistical tests to compare data patterns and validate relationships.


## Tools & Technologies

* **Python Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`
* **Environment:** Jupyter Notebook / VS Code / Anaconda / GitHub


## Key Findings

* There is an **extremely strong positive correlation** between confirmed and death cases (r ≈ 0.99).
* The **p-value ≈ 0**, confirming that this relationship is **statistically significant**.
* The visual trends show that as confirmed cases rise, deaths also increase proportionally, following a consistent linear pattern.

## Next Steps

* Extend the analysis to include **state-level comparisons** within the U.S.
* Build **forecasting models** (e.g., ARIMA or Prophet) to predict future trends.
* Create an **interactive dashboard** for dynamic exploration of the results.


# Goal
1. **Assess the global impact of COVID-19:** Analyze how severely different countries were affected in terms of 2- confirmed cases, deaths, and recovery rates.

2. **Apply Python for real-world data exploration:** Use data analysis tools and visualization techniques to uncover meaningful trends and relationships within real COVID-19 datasets.

3. **Provide actionable public insights:** Translate findings into practical insights that can help identify which regions experienced lower infection and death rates — offering guidance on safer places to live or travel.


# Questions

1. **What were the top five countries with the highest number of confirmed COVID-19 cases between 2020 and 2021?**

2. **What was the cumulative growth of confirmed COVID-19 cases over time globally?**

3. **How did the number of daily confirmed cases compare to daily deaths in the United States during 2021?**

4. **What was the pattern of daily active, recovered, and deceased COVID-19 cases in the United States from 2020 to 2021?**

5. **What was the volatility of daily confirmed and daily death cases in the United States?**

6. **Is there a correlation between confirmed cases, recovered cases, and deaths in the United States during the pandemic period (2020–2021)?**

7. **What is the US COVID-19 Trends of New Cases and Growth Rates**

8. **Has the average growth rate of confirmed cases significantly changed after July 2020?**

9. **Is there a significant relationship between confirmed cases and deaths?**

10. **Can we predict deaths from confirmed cases using a linear regression model?**


# Data Cleaning

Before performing any analysis, the dataset was cleaned and prepared to ensure accuracy and consistency.
The following steps were applied:

```
import ast 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# Set the path to the file you'd like to load
file_path = 'M:/3_datasets/covid_19_data.csv'

# Load the latest version
df = pd.read_csv(file_path)

#clean
df['ObservationDate']= pd.to_datetime(df['ObservationDate'])

df.columns = df.columns.str.replace(' ', '_')
df['Last_Update'] = pd.to_datetime(df['Last_Update'], format='mixed') 
# df['Last_Update'] = pd.to_datetime(df['Last_Update'], format='%m/%d/%y %H:%M', errors='coerce')
# we can also use that format but we cant bec our data include more than one time format for instance (1/22/2020 17:00, 2021-05-30)
# so mixed sayes that me have more than one format

df.set_index('SNo', inplace=True)

df['Country/Region']= df['Country/Region'].fillna('else')
df['Province/State']= df['Province/State'].fillna('else')

df[['Confirmed','Deaths','Recovered']] = df[['Confirmed','Deaths','Recovered']].clip(lower=0) # replace -ve values with 0
```

1. **Data Import & Conversion**

   * Loaded the dataset from the specified path:

     ```python
     df = pd.read_csv('M:/3_datasets/covid_19_data.csv')
     ```
   * Converted the `ObservationDate` column to a proper datetime format:

     ```python
     df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
     ```
   * Standardized column names by replacing spaces with underscores:

     ```python
     df.columns = df.columns.str.replace(' ', '_')
     ```

2. **Handling Mixed Date Formats**

   * The `Last_Update` column contained multiple date formats (e.g., `1/22/2020 17:00` and `2021-05-30`).
     To correctly parse all variations, the `format='mixed'` parameter was used:

     ```python
     df['Last_Update'] = pd.to_datetime(df['Last_Update'], format='mixed')
     ```

3. **Indexing**

   * The `SNo` column was set as the dataset index for better data management:

     ```python
     df.set_index('SNo', inplace=True)
     ```

4. **Missing Data Handling**

   * Filled missing values in `Country/Region` and `Province/State` with `'else'` to maintain data consistency:

     ```python
     df['Country/Region'] = df['Country/Region'].fillna('else')
     df['Province/State'] = df['Province/State'].fillna('else')
     ```

5. **Data Validation**

   * Ensured that negative values in `Confirmed`, `Deaths`, and `Recovered` columns were replaced with zero to avoid invalid entries:

     ```python
     df[['Confirmed', 'Deaths', 'Recovered']] = df[['Confirmed', 'Deaths', 'Recovered']].clip(lower=0)
     ```


### **Result**

After cleaning, the dataset was fully standardized:

* All dates are in consistent datetime format.
* No missing country or province names.
* No negative counts in case numbers.
* Data is ready for accurate statistical analysis and visualization.


# The Analysis

## 1. What were the top five countries with the highest number of confirmed COVID-19 cases between 2020 and 2021?

The analysis focused on identifying the countries most affected by COVID-19 based on the total number of confirmed cases reported between 2020 and 2021. The dataset was first filtered to include data up to May 29, 2021, ensuring that each country’s cumulative case count was accurately captured. The data was then grouped by Country/Region and the total number of confirmed cases was summed for each country. After sorting the results in descending order, the top five countries were selected for visualization.

To illustrate the findings, a horizontal bar chart was created using Seaborn, showing the United States, India, Brazil, France, and Turkey as the countries with the highest number of confirmed cases. Each bar represented the total confirmed cases per country, labeled with exact values for clarity. This visualization provided a clear comparison of how the pandemic’s impact varied globally, emphasizing the significant gap between the United States and other nations in total infections.


View my notebook in details here:
[1_EDA_Intro.ipynb](1_EDA_Intro.ipynb)


### Visualize data 

```
sns.set_theme(style="whitegrid")

df_sns= sns.barplot(data=df_plot, x='Confirmed', y='Country/Region', palette='mako')

for index, value in enumerate(df_plot['Confirmed']):
    plt.text(value +50000, index, f'{int(value):,}')

plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
df_sns.set_xlabel('Number of Confirmed Cases')
df_sns.set_ylabel('Countries')
plt.title('Top 5 Countries by Confirmed COVID-19 in (2021-05-29)')
plt.xticks(rotation=45)
df_sns.set_xlim(0, 35000000)

sns.despine()
plt.show()
```


### Results

![visualization](outputs\output1.png)


### Insights

The bar chart illustrates the **top five countries with the highest number of confirmed COVID-19 cases** as of **May 29, 2021**. The data shows how the pandemic affected different regions globally.

* **United States:** Recorded the highest number of confirmed cases, approximately **33.25 million**, indicating the most widespread outbreak among all countries.
* **India:** Reported around **27.89 million** confirmed cases, reflecting the severe impact of the second wave during 2021.
* **Brazil:** Ranked third with about **16.47 million** cases, showing the extensive spread of the virus in South America.
* **France:** Reported around **5.72 million** confirmed cases.
* **Turkey:** Recorded approximately **5.23 million** confirmed cases.

The chart demonstrates a **significant gap** between the United States and other countries, highlighting the scale of infections within the U.S. population. India’s and Brazil’s high numbers also reveal how large populations and healthcare pressures influenced total case counts. France and Turkey, while still heavily affected, experienced relatively fewer infections compared to the top three nations.

Overall, the visualization provides a clear comparison of the global distribution of COVID-19 cases, emphasizing how **infection levels varied widely across different regions of the world**.


## 2. What was the cumulative growth of confirmed COVID-19 cases over time globally?

The analysis focused on understanding how COVID-19 spread worldwide between January 2020 and May 2021 by examining the cumulative growth of confirmed, recovered, and death cases. The dataset contained global daily reports of COVID-19, including the number of confirmed cases, deaths, and recoveries for each country.

To prepare the data, the entries were grouped by date, and the total counts for each category (Confirmed, Deaths, Recovered) were summed to represent the global situation for each day. After aggregating the data, a stacked area plot was created to visualize how these three measures evolved over time.

The red area in the plot represents confirmed cases, showing a steady rise throughout 2020 and a sharp increase during late 2020 and early 2021, reflecting the global spread of the virus. The green area represents recovered cases, which also increased significantly, indicating successful recovery efforts across countries. The blue area, though much thinner, represents deaths, showing a slower but continuous increase.

Overall, the visualization reveals the exponential nature of the pandemic’s growth, the scale of global recoveries, and the ongoing impact of deaths, offering a clear overview of how COVID-19 evolved globally during this period.

View my notebook in details here:
[1_EDA_Intro.ipynb](1_EDA_Intro.ipynb)


### Visualize data


```
sns.set_theme(style="whitegrid")


plt.figure(figsize=(12, 6))
ax = plt.gca()  # Get the current Axes

ax.stackplot(
    global_daily['ObservationDate'],
    global_daily['Confirmed'],
    global_daily['Deaths'],
    global_daily['Recovered'],
    labels=['Confirmed', 'Deaths', 'Recovered'],
    colors=['#ff9999', '#9999ff', "#99ff99"],  
    alpha=0.8
)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

sns.despine()

plt.title('Global Cumulative COVID-19 Growth Over Time', fontsize=14, pad=12)
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend(loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

### Results

![visualization](outputs\output2.png)

### Insights

This plot illustrates the global cumulative growth of COVID-19 cases, deaths, and recoveries from January 2020 to May 2021. The dataset was grouped by date to aggregate the total confirmed, death, and recovered cases worldwide for each day. Using this data, a stacked area chart was created to visually represent how the pandemic evolved over time.

The red area represents confirmed cases, showing a steady rise throughout 2020, followed by a sharp increase during late 2020 and early 2021, indicating the widespread global surge in infections. The green area represents recoveries, which also grew significantly over time, reflecting global efforts to treat and control the disease. The blue area, much thinner in comparison, represents deaths, showing a slower but consistent increase.

Overall, the visualization highlights the exponential nature of the pandemic’s spread, the large number of recoveries achieved, and the global health challenge posed by COVID-19 during this period. The stacked design makes it easy to compare the trends and proportions of confirmed, recovered, and death cases across time.

# What I Learned


# Insights


# Challenges I Faced


# Conclusion





