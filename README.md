# Employee_Retention_Prediction_Project

## Project Overview
This project focuses on analyzing a human resources dataset to understand the factors influencing employee retention and to build a predictive model to identify employees at risk of leaving the company. Employee turnover can be costly for organizations, and proactively identifying potential departures can enable timely interventions.

## Dataset
The dataset used for this analysis is `HR_comma_sep 1.csv`, which contains various employee attributes and a 'left' column indicating whether an employee has left the company (1) or stayed (0).

## Methodology

### 1. Exploratory Data Analysis (EDA)
The initial phase involved a comprehensive exploratory data analysis to uncover patterns, relationships, and key insights into employee retention. Key steps included:
- **Data Information and Statistics**: Examined data types, non-null counts (`df.info()`), and descriptive statistics (`df.describe()`).
- **Missing Values Check**: Confirmed no missing values in the dataset (`df.isnull().sum()`).
- **Target Variable Distribution**: Analyzed the proportion of employees who left vs. stayed, revealing that approximately 23.81% of employees had left.
- **Numerical Feature Analysis**: Compared the mean of numerical features (e.g., `satisfaction_level`, `average_montly_hours`, `time_spend_company`, `Work_accident`, `promotion_last_5years`) between employees who stayed and those who left.
- **Categorical Feature Analysis**: Calculated and visualized the proportion of employees leaving within each `Department` and `salary` category.

### Key EDA Findings:
- **Satisfaction Level**: Employees who left had significantly lower average satisfaction levels.
- **Work Hours & Time in Company**: Those who left tended to work more hours and had spent more time with the company.
- **Promotions**: Lack of promotion in the last 5 years was strongly associated with employees leaving.
- **Work Accident**: Not a direct driver for leaving, as employees who stayed were more likely to have had a work accident.
- **Salary**: A strong predictor; 'low' salary employees had the highest turnover, while 'high' salary employees had the lowest.
- **Department**: 'HR' and 'Accounting' departments showed higher turnover rates, while 'Management' and 'RandD' had lower rates.

### 2. Data Preprocessing for Modeling
- **Feature Selection**: Selected relevant features including `satisfaction_level`, `average_montly_hours`, `time_spend_company`, `Work_accident`, `promotion_last_5years`, `Department`, and `salary`.
- **One-Hot Encoding**: Converted categorical features (`Department`, `salary`) into numerical format using one-hot encoding to make them suitable for the machine learning model.
- **Data Splitting**: Split the dataset into training (70%) and testing (30%) sets to evaluate the model's performance on unseen data.

### 3. Model Building
- **Logistic Regression**: A Logistic Regression model was chosen and initialized for its interpretability and effectiveness in binary classification tasks.
- **Model Training**: The model was trained using the prepared training data (`X_train`, `y_train`).

## Outcomes and Next Steps
- The exploratory data analysis successfully identified several key factors strongly correlated with employee retention. These insights can inform HR strategies to improve employee satisfaction, address workload, review promotion policies, and investigate department-specific issues.
- A Logistic Regression model has been built and trained to predict employee retention.

### Next Steps:
- **Model Evaluation**: The immediate next step is to measure the accuracy and other relevant performance metrics (e.g., precision, recall, F1-score) of the trained model on the test set to understand its predictive capabilities.
- **Further Optimization**: Depending on the model's performance, further optimization techniques such as hyperparameter tuning or exploring other machine learning algorithms could be considered.
- **Feature Engineering**: Creating new features from existing ones might further enhance model performance.

## Challenges Encountered (Potential)
- **Categorical Feature Handling**: Properly converting categorical variables (`Department`, `salary`) into numerical formats without introducing bias or multicollinearity was crucial.
- **Class Imbalance**: The dataset might exhibit class imbalance (more employees staying than leaving), which could affect model performance and may require techniques like oversampling or undersampling during future model refinement.
- **Model Interpretation**: Ensuring the model's predictions and feature importances are interpretable for business stakeholders is an ongoing consideration.
