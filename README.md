# Bank Marketing Classification
This project focuses on building predictive models to determine whether a client is likely to subscribe to a term deposit based on a dataset from a Portuguese banking institution. The goal is to improve the efficiency of marketing campaigns by identifying potential subscribers.

## Dataset
The dataset used in this project is from the UCI Machine Learning repository. It contains information about clients, the last contact of the current marketing campaign, other attributes, and social and economic context attributes.

## Business Objective: 
The business objective is to develop a predictive model that accurately identifies clients likely to subscribe to a term deposit. This will help optimize marketing efforts, reduce costs, and increase the conversion rate of marketing campaigns.

## Project Steps
The project followed a standard machine learning workflow:

**1. Data Understanding and Loading:** Loaded and initially explored the dataset.

**2. Exploratory Data Analysis (EDA):** Performed detailed univariate, bivariate, and multivariate analysis to understand the data distribution, relationships between features, and the relationship with the target variable. Identified missing values, outliers, and class imbalance.

**3. Data Preprocessing and Feature Engineering:** Handled missing values, encoded categorical features, and addressed outliers. Created a preprocessing pipeline using ColumnTransformer.

**4. Train/Test Split:** Split the data into training and testing sets, ensuring stratification for the imbalanced target variable. The last_contact_duration_seconds feature was excluded from the feature set for realistic modeling.

**5. Baseline Model:** Established a baseline performance using a DummyClassifier.

**6. Model Building and Evaluation (Initial):** Built and evaluated several classification models (Logistic Regression, K Nearest Neighbors, Decision Trees, Support Vector Machines, and Random Forest) using default hyperparameters and key metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

**7. Hyperparameter Tuning:** Improved the performance of the models by tuning hyperparameters using GridSearchCV and RandomizedSearchCV, focusing on optimizing the ROC-AUC score.

**8. Model Comparison (Tuned):** Compared the performance of the tuned models using the evaluation metrics. The tuned Random Forest model showed the best performance in terms of ROC-AUC.

**9. Model Finalization and Next Steps:** Saved the best-performing tuned model. Discussed how to make predictions on new data, potential deployment strategies, and areas for further model improvement (more advanced feature engineering, handling class imbalance, exploring other models, etc.).

## Best Performing Model
Based on the evaluation using ROC-AUC as the primary metric (due to the class imbalance), the **Tuned Random Forest Classifier** was identified as the best performing model.

## Code Structure
The project code is organized within the notebook, with distinct sections for:

* Getting Started (loading libraries, initial setup)
* Data Loading and Initial Inspection
* Data Understanding and Cleaning (EDA, handling missing values, duplicates, outliers)
* Problem Understanding and Task Definition
* Train/Test Split
* Feature Engineering and Preprocessing Pipeline Definition
* Baseline Model Establishment
* Model Building and Evaluation (Initial and Tuned)
* Model Comparisons and Analysis
* Saving the Best Model
* Demonstrating Predictions on New Data

## Repository Contents:

  * **data:** Contains the dataset used to build predictive models. [click here for datasets](https://github.com/gethiten/ClassifiersModelling/tree/main/data)
  * **classifiers.ipynb:** Jupyter Notebook containing the code for data exploration, analysis, modelling, evaluation, and visualization. classifiers.ipynb
  * **images:** Folder containing generated visualizations. Images
  * **README.md:** This file provides an overview of the project. README.md
  * **models:** Tuned Random Forest model 'tuned_random_forest_model.joblib' saved to models folder

## Setup and Usage
To run this notebook, you will need Python and the following libraries installed:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* joblib
