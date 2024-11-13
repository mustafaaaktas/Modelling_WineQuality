from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Random Forest
def train_random_forest(X_train, y_train):
    parameters = {
        'n_estimators': [300, 500],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=0)
    rf_cv = RandomizedSearchCV(estimator=rf,
                               param_distributions=parameters,
                               n_iter=10, cv=5)
    rf_cv.fit(X_train, y_train)

    print("Tuned hyperparameters (RF):", rf_cv.best_params_)
    print("Best accuracy from cross-validation (RF):", rf_cv.best_score_)

    return rf_cv.best_estimator_


# Logistic Regression
def train_logistic_regression(X_train, y_train):
    parameters = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10],
        'solver': ['lbfgs']
    }

    logreg = LogisticRegression(max_iter=1000, random_state=0)
    logreg_cv = RandomizedSearchCV(estimator=logreg,
                                   param_distributions=parameters,
                                   n_iter=10, cv=5)
    logreg_cv.fit(X_train, y_train)

    print("Tuned hyperparameters (LR):", logreg_cv.best_params_)
    print("Best accuracy from cross-validation (LR):", logreg_cv.best_score_)

    return logreg_cv.best_estimator_


# Support Vector Classifier
def train_svc(X_train, y_train):
    # Define parameter grid for tuning
    cv_strategy = StratifiedKFold(n_splits=40)
    parameters = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': [8, 10, 12],  # Kernel coefficient
        'kernel': ['rbf']  # Radial Basis Function kernel
    }

    # Initialize and tune SVC model
    svc = SVC(probability=True)
    svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=cv_strategy)
    svc_cv.fit(X_train, y_train)

    print("Tuned hyperparameters (SVC):", svc_cv.best_params_)
    print("Best accuracy from cross-validation (SVC):", svc_cv.best_score_)

    return svc_cv.best_estimator_


# Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    # Define parameter grid for tuning
    parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(4, 30, 2))
    }

    # Initialize and tune Decision Tree model
    tree = DecisionTreeClassifier(random_state=0)
    tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=5)
    tree_cv.fit(X_train, y_train)

    print("Tuned hyperparameters (Decision Tree):", tree_cv.best_params_)
    print("Best accuracy from cross-validation (Decision Tree):",
          tree_cv.best_score_)

    return tree_cv.best_estimator_


# KNN
def train_knn(X_train, y_train):
    # Define the parameter grid for tuning
    parameters = {
        'n_neighbors': list(range(3, 21, 2)), # Testing odd values from 3 to 20
        'weights': ['uniform', 'distance'], # Uniform vs. distance-based weight
        'p': [1, 2]   # Distance metric: p=1 (Manhattan), p=2 (Euclidean)
    }

    # Initialize and tune KNN model
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(estimator=knn, param_grid=parameters, cv=5)
    knn_cv.fit(X_train, y_train)

    print("Tuned hyperparameters (KNN):", knn_cv.best_params_)
    print("Best accuracy from cross-validation (KNN):", knn_cv.best_score_)

    return knn_cv.best_estimator_


# Gaussian Naive Bayes
def train_gaussian_nb(X_train, y_train):
    # Initialize and train the Gaussian Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Return the trained model
    return gnb
