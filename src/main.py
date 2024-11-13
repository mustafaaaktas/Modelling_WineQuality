from config import *
from data_loading import load_data, data_overview
from visualization import (evaluate_model, plot_distributions,
                           plot_correlation_heatmap,
                           plot_pairplot)
from preprocessing import (categorize_quality,
                           normalize_features,
                           split_data)
from modeling import (train_random_forest,
                      train_logistic_regression,
                      train_svc,
                      train_decision_tree,
                      train_knn,
                      train_gaussian_nb)


file_path = \
    'data/winequality-red.csv'


data = load_data(file_path)
data_overview(data)

# Plot distributions for each feature
plot_distributions(data)

# Plot correlation heatmap
plot_correlation_heatmap(data)

# Plot pairplot
plot_pairplot(data)


# Categorize quality levels
data = categorize_quality(data)
print(data['quality'].value_counts())  # Optional: Distribution of categories


# Step 1: Normalize features and define X, y
X, y = normalize_features(data)


# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Training Models
# Step 3a: Train the Random Forest model
rf_model = train_random_forest(X_train, y_train)

# Step 3b: Evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Test Accuracy:", rf_model.score(X_test, y_test))


# Step 4a: Train Logistic Regression model with hyperparameter tuning
logreg_model = train_logistic_regression(X_train, y_train)

# Step 4b: Evaluate the Logistic Regression model
y_pred_logreg = logreg_model.predict(X_test)
print("Logistic Regression Test Accuracy:", logreg_model.score(X_test, y_test))


# Step 5a: Train and Tune SVC Model
svc_model = train_svc(X_train, y_train)

# Step 5b: Evaluate the SVC Model on Test Set
y_pred_svc = svc_model.predict(X_test)
svc_score = round(svc_model.score(X_test, y_test), 3)
print("SVC Test Accuracy:", svc_score)


# Step 6a: Train and Tune Decision Tree Classifier
tree_model = train_decision_tree(X_train, y_train)

# Step 6b: Evaluate the Decision Tree Model on Test Set
y_pred_tree = tree_model.predict(X_test)
tree_score = round(tree_model.score(X_test, y_test), 3)
print("Decision Tree Test Accuracy:", tree_score)


# Step 7a: Train and Tune K-Neighbors Classifier
knn_model = train_knn(X_train, y_train)

# Step 7b: Evaluate the KNN Model on Test Set
y_pred_knn = knn_model.predict(X_test)
knn_score = round(knn_model.score(X_test, y_test), 3)
print("K-Neighbors Classifier Test Accuracy:", knn_score)


# Step 8a: Train Gaussian Naive Bayes
gnb_model = train_gaussian_nb(X_train, y_train)

# Step 8b: Evaluate the Gaussian Naive Bayes Model on Test Set
y_pred_gnb = gnb_model.predict(X_test)
gnb_score = round(gnb_model.score(X_test, y_test), 3)
print("Gaussian Naive Bayes Test Accuracy:", gnb_score)


# Step Evaluate:
# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

# Evaluate Logistic Regression
y_pred_logreg = logreg_model.predict(X_test)
evaluate_model(y_test, y_pred_logreg, model_name="Logistic Regression")

# Evaluate SVC
y_pred_svc = svc_model.predict(X_test)
evaluate_model(y_test, y_pred_svc, model_name="SVC")

# Evaluate Decision Tree
y_pred_tree = tree_model.predict(X_test)
evaluate_model(y_test, y_pred_tree, model_name="Decision Tree")

# Evaluate KNN (K-Neighbors Classifier)
y_pred_knn = knn_model.predict(X_test)
evaluate_model(y_test, y_pred_knn, model_name="K-Neighbors Classifier")

# Evaluate GNB (Gaussian Naive Bayes)
y_pred_gnb = gnb_model.predict(X_test)
evaluate_model(y_test, y_pred_gnb, model_name="Gaussian Naive Bayes")


# # Results to compare which model did the best (Spoiler Alert! "SVC")
# # Hard-coded results from the output you shared
# results = pd.DataFrame({
#     'Model': [
#         'Random Forest',
#         'Logistic Regression',
#         'SVC',
#         'Decision Tree',
#         'K-Neighbors',
#         'Gaussian Naive Bayes'
#     ],
#     'Cross-Validation Accuracy': [
#         0.862,   # Random Forest
#         0.834,   # Logistic Regression
#         0.849,   # SVC
#         0.832,   # Decision Tree
#         0.861,   # K-Neighbors
#         None     # Gaussian Naive Bayes has no CV value
#     ],
#     'Test Accuracy': [
#         0.8775,  # Random Forest
#         0.855,   # Logistic Regression
#         0.873,   # SVC
#         0.825,   # Decision Tree
#         0.86,    # K-Neighbors
#         0.762    # Gaussian Naive Bayes
#     ]
# })
#
# # Plotting the test accuracy for visual comparison
# sns.set_palette("Purples")
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Test Accuracy', data=results)
# plt.ylim(0.75, 0.9)  # Adjust the y-axis to focus on the range of accuracies
# plt.title("Model Comparison - Test Accuracy")
# plt.xticks(rotation=45)
# plt.show()
#
# # Display the summary table
# print("Model Performance Summary:")
# print(results.sort_values(by="Test Accuracy", ascending=False))

