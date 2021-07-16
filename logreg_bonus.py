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
    m = len(x)
    return (1 / m) * -y.transpose().dot(np.log(h)) - (1 - y).transpose().dot(np.log(1 - h))


def stochastic_gradient_descent(df: pd.DataFrame, features: list, class_index: list):
    """
    Calculate the thetas for each type of the class that's being classified.
    We have as many columns as features.
    Theta is initialized to 0.
    """
    print("Using {} features for the logistic regression".format(len(features)))
    # Define parameters
    thetas = []
    costs = []
    alpha = 0.3
    iterations = 5000
    np_df = df[features].to_numpy()
    rows, columns = np_df.shape
    X = np.hstack((np.ones((rows, 1)), np_df))
    m = len(X)
    classes = df[class_index].unique()
    set_length = len(X)
    decay = 0.1
    batch_size = int(set_length / 10)
    # Calculate a different theta for each classes
    for i in classes:
        # Replace current house name by 1
        # and set all other to 0 (one vs all)
        print("Finding theta for {}".format(i))
        y = np.where(df[class_index] == i, 1, 0)
        theta = np.zeros(columns + 1)
        current_cost = []
        diff = 0
        for i in range(iterations):
            # Generate a random batch of batch_size
            current_indexes = np.unique(np.random.randint(set_length, size=batch_size))
            batch_x = X[current_indexes, :]
            batch_y = y[current_indexes]
            h = sigmoid(batch_x, theta)
            gradient = (1 / m) * batch_x.transpose().dot(h - batch_y)
            # Calculate the diff between the previous gradient and apply decay
            diff = decay * diff - alpha * gradient
            theta += diff
            current_cost.append(cost(batch_x, batch_y, theta))
        thetas.append(theta)
        costs.append(current_cost)
    # Cost plot
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
    predictions = np.array([sigmoid(X, theta) for theta in thetas])
    for index, column in enumerate(predictions.transpose()):
        classified[index] = column.argmax()
    return classified.astype(int)


def accuracy(df: pd.DataFrame, results: np.ndarray, y: np.ndarray, classes: list):
    """
    Calculate the accuracy of the predictions with a given set of results by comparing each results one by one.
    """
    indexes = {class_name: index for index, class_name in enumerate(classes)}
    accuracy = 0
    for i, result in enumerate(y):
        if indexes[result] == results[i]:
            accuracy += 1
    return (accuracy / len(df)) * 100


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
    df.dropna(inplace=True, axis=0)
    class_column = "Hogwarts House"
    features = [
        # "Arithmancy",
        # "Potions",
        # "Care of Magical Creatures",
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
    normalized = normalize(df, features)
    thetas = stochastic_gradient_descent(normalized, features, class_column)
    classified = classify(normalized, features, thetas)
    classes = df[class_column].unique()
    print("Accuracy: {:.2f}%".format(accuracy(df, classified, df[class_column], classes)))
    # Save thetas -- One classification per column
    thetas_dict = {classes[index]: theta for index, theta in enumerate(thetas)}
    thetas_dict["Features"] = features.copy()
    thetas_dict["Features"].insert(0, "Intercept")
    df_thetas = pd.DataFrame(thetas_dict)
    try:
        df_thetas.to_csv("thetas.csv", index=False)
        print("Saved thetas to `thetas.csv`")
    except IOError as err:
        print("Failed to save thetas: {}".format(err))
        exit(1)
