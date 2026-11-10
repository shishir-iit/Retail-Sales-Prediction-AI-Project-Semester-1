# Retail-Sales-Prediction-AI-Project-Semester-1
# 📊 Retail Sales Forecasting Using Linear Regression

## 1. Project Overview

The objective of this project is to develop a Machine Learning model that predicts the next day's sales quantity using historical online retail transaction data.

The project uses **Linear Regression**, a supervised machine learning algorithm, to identify patterns in historical sales and make future predictions.

The project involves cleaning and preprocessing transaction records, aggregating sales data by date, extracting useful features, training a Linear Regression model, and evaluating its prediction performance.

## 2. Problem Statement

Retail businesses need to estimate future sales to manage inventory, plan stock requirements, and make better business decisions.

This project addresses the problem by developing a predictive model that uses historical daily sales information to forecast sales for the following day.

## 3. Objectives

- Understand and preprocess historical retail transaction data.
- Remove duplicate, cancelled, invalid, and extreme transaction records.
- Convert transaction-level data into daily sales data.
- Extract meaningful features from historical sales.
- Implement a Linear Regression model for next-day sales prediction.
- Evaluate model performance using suitable evaluation metrics.
- Visualize actual and predicted sales.

## 4. Dataset Description

**Dataset:** Online Retail Dataset

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains historical transactions from a UK-based online retail store.

Important columns include:

| Column | Description |
|---|---|
| InvoiceNo | Transaction invoice number |
| StockCode | Product identification code |
| Description | Product description |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Customer identification |
| Country | Customer's country |

The target variable in this project is **next-day total sales value**, measured in GBP (£), rather than the number of individual products sold.

## 5. Technologies and Libraries Used

- Python
- Pandas – Data cleaning and manipulation
- NumPy – Numerical computations
- Matplotlib – Data visualization
- Scikit-learn – Machine Learning and model evaluation
- Jupyter Notebook – Project implementation

## 6. Methodology

The project follows these steps:

**Step 1: Data Collection**

Download the Online Retail dataset from the UCI Machine Learning Repository.

**Step 2: Data Preprocessing**

- Remove duplicate transactions.
- Convert InvoiceDate into datetime format.
- Remove cancelled invoices and returns.
- Remove transactions with zero or negative prices.
- Remove non-product transaction codes.
- Filter extreme transaction values.

**Step 3: Daily Sales Aggregation**

Calculate daily sales by multiplying Quantity by UnitPrice and aggregating the values for each date.

**Step 4: Feature Engineering**

Create the following features:

- Sales: Current day's sales.
- roll3: Average sales over the last 3 days.
- roll7: Average sales over the last 7 days.
- lag7: Sales from 6 days earlier.
- next_sat: Indicates whether the next day is Saturday.
- next_sun: Indicates whether the next day is Sunday.
- q4: Indicates whether the current date falls in October, November, or December.

**Step 5: Model Training**

Use Linear Regression from Scikit-learn to train the prediction model. Split the dataset into 80% training data and 20% testing data.

**Step 6: Model Evaluation**

Evaluate the model using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) Score

**Step 7: Sales Prediction**

Use the trained model and the latest available sales information to predict the next day's sales.

## 7. Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

It establishes a relationship between input features and the target variable.

The general equation is:

`Y = b0 + b1X1 + b2X2 + ... + bnXn`

Where:

- Y = Predicted next-day sales
- b0 = Intercept
- b1, b2, ..., bn = Model coefficients
- X1, X2, ..., Xn = Input features

In this project, Linear Regression predicts the next day's sales using historical sales patterns and calendar-based features.

## 8. Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| MAE | Measures the average absolute prediction error |
| RMSE | Measures prediction error while penalizing larger errors |
| R² Score | Measures how well the model explains variations in sales |

A higher R² score generally indicates better model performance.

## 9. Project Structure

```text
Retail-Sales-Forecasting/
│
├── data/Online_Retail.csv
├── src/main.py
├── src/model.py
├──outputs/
├── requirements.txt
├── README.md
```

## 10. Installation and Setup

### Prerequisites

- Python 3.10 or above
- Jupyter Notebook
- Git

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Navigate to the Project Folder

```bash
cd Retail-Sales-Forecasting
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the Online Retail dataset from the UCI Machine Learning Repository and place `Online_Retail.csv` in the project folder.

### Step 5: Run the Notebook

```bash
jupyter notebook
```

Open `Next_Day_Sales_Linear_Regression.ipynb` and execute all cells in order.

**Note:** Update the dataset loading path in the notebook to `Online_Retail.csv` when running locally instead of using the Google Colab path `/content/Online_Retail.csv`.

## 11. Expected Output

After successfully executing the notebook, the project produces:

1. Cleaned and preprocessed retail transaction data.
2. Daily sales aggregation.
3. Daily sales trend visualization.
4. Extracted features for prediction.
5. Trained Linear Regression model.
6. Model evaluation results (MAE, RMSE and R²).
7. Actual vs Predicted Sales graph.
8. Prediction of next-day sales in GBP (£).

## 12. Results and Visualization

The project generates visualizations to understand historical sales trends and evaluate prediction performance.

**Visualization 1: Daily Sales Trend**

Shows the changes in daily sales over the available dataset period.
<img width="771" height="290" alt="Screenshot 2026-11-10 204326" src="https://github.com/user-attachments/assets/c52e14f2-8604-434c-8b3e-3af9a02bd294" />


**Visualization 2: Actual vs Predicted Sales**

Compares the actual next-day sales values with the values predicted by the Linear Regression model.

<img width="474" height="404" alt="Screenshot 2026-11-10 203214" src="https://github.com/user-attachments/assets/878fa71d-054e-4f69-9e9e-6c388084c8d3" />


## 13. Team Members and Responsibilities

| Team Member | Name | Entry Number | Role |
|---|---|---|---|
| Member 1 | Varsha Yadv | 2026ESB1498 | Documentation Lead |
| Member 2 | Mohammed Rizwan N |2026EEbB1380 | Execution Lead |
| Member 3 | Vijval Bhoriya | 2026AIB1096 | PPT Lead |
| Member 4 | Ananthaneni Naga Shishir | 2026MCB1509 | Video Lead |

All team members contributed to the project through coding, testing, documentation, and collaboration.

## 14. Project Deliverables

- GitHub repository containing the project code and documentation.
- Python requirements file.
- Project presentation (5 slides).
- Project demonstration video (maximum 5 minutes).

**Presentation:** [Add PPT link here]

**Demonstration Video:** [Add video link here]

## 15. AI-Use Disclosure

AI tools were used to assist with project development and documentation.

- ChatGPT: Assisted with understanding concepts, coding support, debugging, and README documentation.
- All generated code and outputs should be reviewed and tested by the project team.
- The team is responsible for understanding the implementation, verifying the results, and explaining the project.

*Update this disclosure to accurately reflect the AI tools and tasks actually used by your team.*

## 16. Conclusion

This project demonstrates how Linear Regression can be applied to historical retail transaction data to forecast next-day sales. Through data preprocessing, feature engineering, model training, and evaluation, the project provides an introduction to practical Machine Learning applications in retail business analytics.

The project also highlights the importance of historical sales patterns and calendar-based features in understanding and predicting future sales.
