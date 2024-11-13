# Red Wine Quality Classification

This project aims to classify the quality of red wine samples based on various chemical properties using several machine learning models. The dataset consists of multiple features (such as acidity, sulfur dioxide levels, and alcohol content) that influence wine quality, classified into different quality categories. The objective is to determine the best-performing model for accurate wine quality classification.

## Project Structure

The project is organized into the following modules:

- **`data_loading.py`**: Responsible for importing and preprocessing the dataset.
- **`import.py`**: Handles importing essential libraries and dependencies.
- **`preprocessing.py`**: Contains data normalization and scaling functions.
- **`modeling.py`**: Defines and trains multiple classification models.
- **`visualization.py`**: Generates plots, including confusion matrices and classification reports, to evaluate model performance.
- **`config.py`**: Stores configuration parameters for easy adjustment of key variables (e.g., test size split).
- **`main.py`**: The main script that executes the workflow from data loading to model evaluation.

## Project Workflow

### 1. **Data Loading**
- The dataset was loaded and initial statistical checks were performed using functions in `data_loading.py`.
- Key checks included verifying column names, checking for missing values, and obtaining a summary of each feature (mean, median, min, max).
- **Exploratory Data Analysis (EDA)**: This phase included exploring feature distributions, identifying correlations, and ensuring data quality.

### 2. **Data Categorization**
- The `quality` column (representing wine quality on a scale of 3 to 8) was categorized into three classes:
    - **Good**: Quality scores of 8 and 7.
    - **Middle**: Quality scores of 6 and 5.
    - **Bad**: Quality scores of 4 and below.
- These categories provided a clearer basis for classification than using the raw quality score.

### 3. **Data Normalization**
- Feature normalization was applied using min-max scaling, defined as:
  `X_scaled = (X - X_min) / (X_max - X_min)`
- Normalization helped reduce the influence of feature scale, making the models more sensitive to relative feature importance.

### 4. **Data Splitting**
- The dataset was split into **training** (75%) and **test** (25%) sets to evaluate model performance.
- This split was controlled using a consistent `random_state` to ensure reproducibility.

### 5. **Model Selection**
- Multiple classification models were trained and evaluated to find the best-performing model:
    - **Random Forest Classifier**
    - **Logistic Regression**
    - **Support Vector Classifier (SVC)**
    - **Decision Tree Classifier**
    - **K-Neighbors Classifier**
    - **Gaussian Naive Bayes**
- For each model, hyperparameters were tuned using `GridSearchCV` and `RandomizedSearchCV` to optimize accuracy.

### 6. **Model Evaluation**
- Each model’s performance was evaluated using cross-validation accuracy and test accuracy.
- Evaluation metrics included:
    - **Confusion Matrix**: To visualize misclassifications.
    - **Classification Report**: Providing precision, recall, and F1-score for each quality category (Good, Middle, Bad).
- Among the tested models, **Support Vector Classifier (SVC)** with an RBF kernel yielded the best performance across cross-validation and test accuracy.

### 7. **Final Model Selection**
- After evaluating all models, SVC was selected as the final model due to its superior performance.
- Further parameter tuning was applied to SVC (`C`, `gamma`), and cross-validation accuracy was fine-tuned to improve generalization and reduce overfitting.

### 8. **Results and Observations**
- SVC was able to capture patterns effectively within the "Good" quality category, though the "Middle" and "Bad" categories required careful balancing.
- Class imbalance was an identified challenge, as the dataset contained a majority of "Good" wines. Different techniques like class weighting were explored, though tuning `C` and `gamma` without class weights yielded optimal results.

---
