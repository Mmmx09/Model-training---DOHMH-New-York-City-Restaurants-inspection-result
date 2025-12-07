# NYC Restaurant Inspection Result Prediction

## Project Overview

This project uses machine learning techniques to analyze hygiene inspection records from the New York City Department of Health and Mental Hygiene (DOHMH). 

The primary objective is to **predict the result of a restaurant's next hygiene inspection** based on historical data, location, and cuisine characteristics. This tool aims to help identify high-risk establishments and understand the characteristics of restaurant violations across the city.

## Dataset Description

The dataset is sourced from **NYC Open Data** and represents a comprehensive record of restaurant inspections in New York City.

* **Source:** [DOHMH New York City Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data)
* **Provider:** NYC Department of Health and Mental Hygiene (DOHMH)
* **Size:** Approximately 800,000 rows and 26 columns.
* **Scope:** Includes every sustained or not yet adjudicated violation citation from active restaurants up to three years prior to the data pull date.

### Key Features
* **Identifiers:** `CAMIS` (Unique Record ID), `DBA` (Restaurant Name).
* **Location:** `BORO` (Administrative District), `Latitude`, `Longitude`, `Zip Code`.
* **Details:** `CUISINE DESCRIPTION`, `INSPECTION DATE`.
* **Outcome:** `ACTION`, `VIOLATION CODE`, `violation_description`, `SCORE`, `GRADE`.

> **Data Note:** Establishments with an inspection date of `1/1/1900` represent new businesses that have not yet received an inspection. These records were handled specifically during the data preprocessing stage.

**Attachment:** 

[Restaurant Inspection Data Dictionary.xlsx](RestaurantInspectionDataDictionary_09242018.xlsx)

[About NYC Restaurant Inspection Data on NYC OpenData.docx](About_NYC_Restaurant_Inspection_Data_on_NYC_OpenData_050222.docx)

## Project Goal

Project goal is to build a predictive model that can find the outcome of a future inspection: (Regression problem)
* **Input:** Historical inspection history, restaurant metadata (Cuisine, violation description), and actions.
* **Output:** The predicted result of the next inspection score.

## Methodology

### 1. Data Preprocessing
* **Delete useless columns:** We delete `GRADE DATE`, `GRADE`, `RECORD DATE`.
> **Delete reason:** From the official documents, it can be known that the “GRADE” is derived based on the size of the “SCORE”. We can also predict the score to obtain the grade more accurately. Therefore, I choose to delete the grade column (at the same time, grade has 51% missing values, which will affect the result).
<img src="images/GRADE.png" width="600" alt="EDA Chart">

* **Time characteristics:** Seperate `INSPECTION DATE` to `year`, `month`, `weekday`.
* **Cleaning:** Using the `Processe_df` function to fill the missing values.

### 2. Exploratory Data Analysis (EDA)
* Analyzed the distribution of inspection grades across different NYC boroughs.
* Investigated correlations between specific cuisine types and violation rates.

### 3. Add feature values
In the first round of random forest model training, I found that the value of **$R^2$** was stuck at around 0.5 and could not be improved (There is even the problem of overfitting.). 

<img src="images/first round random forest model result.png" width="200" alt="result Chart">

So in order to enhance the model's expressiveness, I chose to incorporate two related feature values:
1. Calculate the number of days since the last inspection.
2. Calculate the average of the scores of the past three times.

### 4. Modeling
We evaluated several machine learning algorithms to determine the best predictor:
* **[Model 1]** Random Forest model
* **[Model 2]** XGBoost model
* **[Model 3]** CatBoost model
* **[Model 4]** GradientBoost model
  
## Results & Evaluation

The best performing model was **[Random Forest model]**, achieving the following performance on the test set:

| Metric | Score |
| :--- | :--- |
| **$R^2$** | **0.80** |

**Key Findings:**

#### 1. Model Performance & Selection

| Model (**$R^2$**) | Score on Training set |  Score on validation set |
| :--- | :--- |:--- |
| Random forest model | **0.819** | **0.800** |
| XG boost model | **0.843** | **0.811** |
| Cat boost model | **0.813** | **0.805** |
| Gradient boost model | **0.849** | **0.811** |

Overall, I found that the random forest model has better expressiveness and stronger generalization ability. The expressiveness of other models is slightly inferior to that of Random Forest, and they all have a slight overfit. Regarding the upper limit of this model, I found that under various algorithms, the **$R^2$** of the model is around 0.80-0.82, which is difficult to improve and is currently the optimal value under feature engineering.

#### 2.Feature Importance Insights

Through the analysis of importance and feature correlation, I discovered the three core elements for predicting the hygiene score of restaurants in New York City:
* `avg_last_3_scores`: The average score of the past three checks (Historical inertia).
* `days_since_last`: The number of days since the last inspection (Regulatory pressure and time decay).
* `action`: Administrative handling result.
