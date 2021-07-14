import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray, theta: np.ndarray):
    """
    The "boundary function",
    which determines to which group the given z is supposed to be in.
    """
    return 1 / (1 + np.exp(-x.dot(theta)))


def cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Cost of the given x with the weights theta against the given answers y.
    """
    h = sigmoid(x, theta)
    return -(1 / len(x)) * np.sum(y * np.log(h) + (1 - y) * (np.log(1 - h)))


def gradient_descent(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Calculate the next thetas for the current given thetas.
    It's one iteration of the gradient descent.
    """
    m = len(x)
    h = sigmoid(x, theta)
    return (1 / m) * x.transpose().dot(h - y)


def logistic_regression(df: pd.DataFrame, features: list, idx: list):
    """
    Calculate the thetas for each type of the class that's being classified.
    We have as many columns as features.
    Theta is initialized to 0.
    """
    print("Using {} features for the logistic regression".format(len(features)))
    thetas = []
    costs = []
    alpha = 0.1
    iterations = 2500
    np_df = df[features].to_numpy()
    rows, columns = np_df.shape
    X = np.hstack((np.ones((rows, 1)), np_df))
    classes = df[idx].unique()
    for i in classes:
        # Replace current house name by 1
        # and set all other to 0 (one vs all)
        print("Finding theta for {}".format(i))
        y = np.where(df[idx] == i, 1, 0)
        theta = np.zeros(columns + 1)
        current_cost = []
        for i in range(iterations):
            theta -= alpha * gradient_descent(X, y, theta)
            current_cost.append(cost(X, y, theta))
        thetas.append(theta)
        costs.append(current_cost)
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
        min = data.min()
        max = data.max()
        normalized[name] = normalized[name].apply(minMaxNormalize, args=(min, max))
    return normalized


def predict(x: np.ndarray, theta: np.ndarray):
    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return sigmoid(X, theta)


def classify(df: pd.DataFrame, features: list, thetas: np.ndarray, idx: list):
    np_df = df[features].to_numpy()
    rows, columns = np_df.shape
    classified = np.zeros(rows)
    X = np.hstack((np.ones((rows, 1)), np_df))
    predictions = np.array([sigmoid(X, theta) for theta in thetas])
    for index, column in enumerate(predictions.transpose()):
        classified[index] = column.argmax()
    return classified.astype(int)


def accuracy(df: pd.DataFrame, results: np.ndarray, idx: list):
    classes = df[idx].unique()
    indexes = {class_name: index for index, class_name in enumerate(classes)}
    accuracy = 0
    for i, result in enumerate(df[idx]):
        if indexes[result] == results[i]:
            accuracy += 1
    return (accuracy / len(df)) * 100


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 2:
        print("Usage: scatter_plot.py [dataset.csv]")
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
    df.dropna(inplace=True, axis=0)
    class_column = "Hogwarts House"
    features = [
        # "Arithmancy",
        # "Potions",
        # "Care of Magical Creatures",
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
    df["Best Hand"] = df["Best Hand"].astype(str).map({"Left": 0.0, "Right": 1.0})
    normalized = normalize(df, features)
    thetas = logistic_regression(normalized, features, class_column)
    classified = classify(normalized, features, thetas, class_column)
    print("Accuracy: {:.2f}%".format(accuracy(df, classified, class_column)))
    classes = df[class_column].unique()
    df_thetas = pd.DataFrame({classes[index]: theta for index, theta in enumerate(thetas)})
    try:
        df_thetas.to_csv("thetas.csv")
        print("Saved thetas to `thetas.csv`")
    except IOError as err:
        print("Failed to save thetas: {}".format(err))
        exit(1)
