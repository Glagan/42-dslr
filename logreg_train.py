import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray):
    """
    The "boundary function",
    which determines to which group the given z is supposed to be in.
    """
    return 1 / (1 + np.exp(-z))


def cost(x: np.ndarray, y: np.ndarray, m: int, theta: np.ndarray):
    """
    Cost of the given x with the weights theta against the given answers y.
    """
    h = sigmoid(x.dot(theta))
    return -np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h))) / m


def gradient_descent(
    df: pd.DataFrame,
    class_column: str,
    features: list,
    alpha: int = 0.025,
    iterations: int = 1000,
    verbose: bool = True,
):
    """
    Calculate the thetas for each type of the class that's being classified.
    We have as many columns as features.
    Thetas are initialized to 0.
    """
    if verbose:
        print(f"Using {len(features)} features for the logistic regression")
    # Define parameters
    thetas = []
    costs = []
    x = df[features].to_numpy()
    rows, columns = x.shape
    X = np.hstack((np.ones((rows, 1)), x))
    xT = X.transpose()
    m = len(X)
    classes = df[class_column].unique()
    # Calculate a different theta for each classes
    step = iterations / 10
    for i in classes:
        # Replace current house name by 1
        # and set all other to 0 (one vs all)
        if verbose:
            print(f"Finding theta for {i}")
        y = np.where(df[class_column] == i, 1, 0)
        theta = np.zeros(columns + 1)
        current_cost = []
        for j in range(iterations):
            h = sigmoid(X.dot(theta))
            gradient = (1 / m) * xT.dot(h - y)
            theta -= alpha * gradient
            if verbose:
                current_cost.append(cost(X, y, m, theta))
                if j % step == 0:
                    current_accuracy = accuracy(sigmoid(X.dot(theta)) >= 0.5, y, [0, 1])
                    print(f"it={j}, cost={current_cost[-1]:.2f}, accuracy={current_accuracy:.2f}")
        thetas.append(theta)
        if verbose:
            costs.append(current_cost)
    # Cost plot
    if verbose:
        row_cols = int(len(costs) / 2)
        fig, axs = plt.subplots(row_cols, row_cols)
        flat_axs = axs.flatten()
        i = 0
        for class_cost in costs:
            flat_axs[i].set_title(classes[i])
            flat_axs[i].set_ylabel("Cost")
            flat_axs[i].set_xlabel("Iterations")
            flat_axs[i].plot(class_cost)
            i += 1
        fig.suptitle("Cost of all classification over Iterations")
        plt.show()
    return thetas


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
    """
    Make a prediction for each thetas for each row in the DataFrame.
    The highest score from all of the predictions is the class to which it belongs to.
    The binary prediction is then converted to the original length of the classes
    (found with the index of the theta which generated the highest score).
    """
    np_df = df[features].to_numpy()
    rows, columns = np_df.shape
    classified = np.zeros(rows)
    X = np.hstack((np.ones((rows, 1)), np_df))
    predictions = np.array([sigmoid(X.dot(theta)) for theta in thetas])
    for index, column in enumerate(predictions.transpose()):
        classified[index] = column.argmax()
    return classified.astype(int)


def accuracy(predictions: np.ndarray, y: np.ndarray, classes: list):
    """
    Calculate the accuracy of the predictions with a given set of results by comparing each results one by one.
    """
    indexes = {class_name: index for index, class_name in enumerate(classes)}
    accuracy = 0
    for i, result in enumerate(y):
        if indexes[result] == int(predictions[i]):
            accuracy += 1
    return accuracy / len(predictions)


def train(df: pd.DataFrame, class_column: str, features: list, verbose: bool = True) -> pd.DataFrame:
    return gradient_descent(df, class_column, features, verbose=verbose)


def cleanDataset(df: pd.DataFrame, features: list = []) -> pd.DataFrame:
    df.fillna(method="ffill", inplace=True, axis=0)
    # Convert string features to int
    if len(features) == 0:
        features = df.columns.to_list()
    for feature in features:
        if df[feature].dtype == "object":
            df[feature], _ = df[feature].factorize()
    return normalize(df, features)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 2:
        print("Usage: logreg_train.py [dataset.csv]")
        exit()
    dataset = "datasets/dataset_train.csv"
    if argc == 2:
        dataset = sys.argv[1]
    try:
        df = pd.read_csv(dataset)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
        exit(1)
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
        exit(1)
    # Delete NA columns since they have missing data
    class_column = "Hogwarts House"
    features = [
        # "Arithmancy",
        # "Care of Magical Creatures",
        # "Potions",
        "Best Hand",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        "Charms",
        "Divination",
        "Muggle Studies",
        "History of Magic",
        "Transfiguration",
        "Flying",
    ]
    # Normalize and calculate thetas for each classes
    normalized = cleanDataset(df, features)
    thetas = train(normalized, class_column, features, verbose=True)
    classified = classify(normalized, features, thetas)
    classes = df[class_column].unique()
    print(f"Accuracy: {accuracy(classified, df[class_column], classes):.2f}")
    # Construct result DataFrame -- One classification per column
    thetas_dict = {classes[index]: theta for index, theta in enumerate(thetas)}
    thetas_dict["Features"] = features.copy()
    thetas_dict["Features"].insert(0, "Intercept")
    df_thetas = pd.DataFrame(thetas_dict)
    # Save thetas -- One classification per column
    try:
        df_thetas.to_csv("thetas.csv", index=False)
        print("Saved thetas to `thetas.csv`")
    except IOError as err:
        print("Failed to save thetas: {}".format(err))
        exit(1)
