import sys
import pandas as pd
from logreg_train import normalize, classify


def predict(df: pd.DataFrame, features: list, classes: list, thetas: pd.DataFrame) -> pd.DataFrame:
    classified = classify(df, features, thetas)
    named_classes = [classes[index] for index in classified]
    return pd.DataFrame(named_classes, columns=["Hogwarts House"])


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
        print(f"Could not find or read thetas, use `logreg_train.py` first: {err}")
        exit(1)
    except pd.errors.ParserError as err:
        print(f"Invalid dataset: {err}")
        exit(1)
    # Load, convert and normalize the test Dataset
    try:
        df = pd.read_csv(dataset)
        # Convert string features to int
        for feature in features:
            if df[feature].dtype == "object":
                df[feature], _ = df[feature].factorize()
        df.fillna(method="ffill", inplace=True, axis=0)
        normalized = normalize(df, features)
    except IOError as err:
        print(f"Failed to read dataset: {err}")
        exit(1)
    except pd.errors.ParserError as err:
        print(f"Invalid dataset: {err}")
        exit(1)
    # Predict each values from the test dataset
    df_houses = predict(normalized, features, classes, thetas)
    # Save found houses
    try:
        print(df_houses)
        df_houses.to_csv("houses.csv", index=True, index_label="Index")
        print("Saved predictions to `houses.csv`")
    except IOError as err:
        print(f"Failed to save predictions: {err}")
        exit(1)
