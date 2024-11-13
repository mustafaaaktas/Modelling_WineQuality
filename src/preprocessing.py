import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def categorize_quality(data):
    # Map quality ratings into categories
    quality_mapping = {
        8: 'Good',
        7: 'Good',
        6: 'Middle',
        5: 'Middle',
        4: 'Bad',
        3: 'Bad'
    }
    data['quality'] = data['quality'].map(quality_mapping)
    print("Quality categorized into Good, Middle, and Bad.")
    return data


def normalize_features(data):
    # Separate features and target
    x = data.drop(columns='quality')
    y = data['quality']

    # Apply MinMaxScaler to the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    print("Features normalized to [0, 1] range.")
    return x_scaled, y


def split_data(X, y, test_size=0.25, random_state=0):
    (X_train, X_test, y_train,
     y_test) = train_test_split(X, y,
                                test_size=test_size,
                                random_state=random_state)
    print("Data split into 25% training and 75% testing sets.")
    return X_train, X_test, y_train, y_test
