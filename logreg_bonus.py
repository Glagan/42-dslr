import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logreg_train import sigmoid, cost, classify, accuracy, cleanDataset


def stochastic_gradient_descent(
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
    print(f"Using {len(features)} features for the logistic regression")
    # Define parameters
    thetas = []
    costs = []
    alpha = 0.3
    iterations = 5000
    x = df[features].to_numpy()
    rows, columns = x.shape
    X = np.hstack((np.ones((rows, 1)), x))
    m = len(X)
    classes = df[class_column].unique()
    set_length = len(X)
    decay = 0.1
    batch_size = int(set_length / 10)
    # Calculate a different theta for each classes
    step = iterations / 10
    for i in classes:
        # Replace current house name by 1
        # and set all other to 0 (one vs all)
        print(f"Finding theta for {i}")
        y = np.where(df[class_column] == i, 1, 0)
        theta = np.zeros(columns + 1)
        current_cost = []
        diff = 0
        for j in range(iterations):
            # Generate a random batch of batch_size
            current_indexes = np.unique(np.random.randint(set_length, size=batch_size))
            batch_x = X[current_indexes, :]
            batch_y = y[current_indexes]
            h = sigmoid(batch_x.dot(theta))
            gradient = (1 / m) * batch_x.transpose().dot(h - batch_y)
            # Calculate the diff between the previous gradient and apply decay
            diff = decay * diff - alpha * gradient
            theta += diff
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


def train(df: pd.DataFrame, class_column: str, features: list, verbose: bool = True) -> pd.DataFrame:
    return stochastic_gradient_descent(df, class_column, features, verbose=verbose)


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
