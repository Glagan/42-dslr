import sys
import pandas as pd
import numpy as np


def sigmoid(z: np.ndarray):
    """
    The "boundary function",
    which determines to which group the given z is supposed to be in.
    """
    return 1 / (1 + np.exp(-z))


def minMaxNormalize(val, min, max):
    return (val - min) / (max - min)


def normalize(df: pd.DataFrame, features: list):
    """
    Apply a min-max normalization to all features columns in a Pandas DataFrame.
    """
    normalized = df.copy()
    for (name, data) in normalized[features].iteritems():
        normalized[name] = normalized[name].apply(minMaxNormalize, args=(data.min(), data.max()))
    return normalized


def classify(df: pd.DataFrame, features: list, thetas: np.ndarray):
    np_df = df[features].to_numpy()
    rows, columns = np_df.shape
    classified = np.zeros(rows)
    X = np.hstack((np.ones((rows, 1)), np_df))
    predictions = np.array([sigmoid(X.dot(theta)) for theta in thetas])
    for index, column in enumerate(predictions.transpose()):
        classified[index] = column.argmax()
    return classified.astype(int)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 3:
        print("Usage: logreg_predict.py [dataset.csv] [thetas.csv]")
        exit()
    dataset = "datasets/dataset_test.csv"
    if argc >= 2:
        dataset = sys.argv[1]
    thetas_file = "thetas.csv"
    if argc >= 3:
        thetas_file = sys.argv[2]
    # Read thetas file and get columns
    try:
        df_thetas = pd.read_csv(thetas_file)
        features = df_thetas["Features"].to_list()
        features.remove("Intercept")
        df_thetas.drop(columns=["Features"], inplace=True)
        thetas = df_thetas.to_numpy().transpose()
        classes = df_thetas.columns.to_list()
    except IOError as err:
        print("Could not find or read thetas, use `logreg_train.py` first: {}".format(err))
        exit(1)
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
        exit(1)
    # Predict each values from the test dataset
    try:
        df = pd.read_csv(dataset)
        # Convert string features to int
        for feature in features:
            if df[feature].dtype == "object":
                df[feature], _ = df[feature].factorize()
        normalized = normalize(df, features)
        classified = classify(normalized, features, thetas)
        named_classes = [classes[index] for index in classified]
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
        exit(1)
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
        exit(1)
    # Save found houses
    try:
        df_houses = pd.DataFrame(named_classes, columns=["Hogwarts House"])
        print(df_houses)
        df_houses.to_csv("houses.csv", index=True, index_label="Index")
        print("Saved predictions to `houses.csv`")
    except IOError as err:
        print("Failed to save predictions: {}".format(err))
        exit(1)
