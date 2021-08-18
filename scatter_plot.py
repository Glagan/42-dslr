import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "Ravenclaw": "indianred",
    "Slytherin": "green",
    "Gryffindor": "gold",
    "Hufflepuff": "teal",
}


def show_scatter(df: pd.DataFrame, class_index: str):
    df.set_index("Index", drop=True, inplace=True)
    ncols = int((len(df.columns) / 3) + 0.5)
    grouped = df.groupby(class_index)
    fig, dim_axs = plt.subplots(nrows=3, ncols=ncols, figsize=(16, 8))
    axs = dim_axs.flatten()
    i = 0
    for feature, values in df.items():
        if feature == class_index:
            continue
        for house, y in grouped:
            axs[i].scatter(x=y[feature].index, y=y[feature], marker=".", c=colors[house], label=house)
            axs[i].set_title(feature)
            axs[i].legend()
        i += 1
    while i < len(axs):
        axs[i].axis("off")
        i += 1
    plt.show()


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
        non_number = ["First Name", "Last Name", "Birthday", "Best Hand"]
        df.drop(columns=non_number, inplace=True)
        show_scatter(df, "Hogwarts House")
        print("Similar features between all houses should have the same groups for the same colors.")
        print("> *History of Magic* and *Transfiguration* are similar between all four houses.")
        print("> *Arithmancy* and *Care of Magical Creatures* are also similar between all four houses.")
    except IOError as err:
        print(f"Failed to read dataset: {err}")
    except pd.errors.ParserError as err:
        print(f"Invalid dataset: {err}")
