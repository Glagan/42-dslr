import matplotlib.pyplot as plt
import sys
import pandas as pd


def show_histograms(df: pd.DataFrame, class_index: str) -> None:
    ncols = int(len(df.columns) / 2)
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(16, 8))
    all_axs = axs.flatten()
    group = df.groupby([class_index])
    i = 0
    for feature, values in df.items():
        if feature == class_index:
            continue
        group[feature].hist(alpha=0.5, ax=all_axs[i], legend=True)
        all_axs[i].set_title(feature)
        i += 1
    while i < len(all_axs):
        all_axs[i].axis("off")
        i += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 2:
        print("Usage: histogram.py [dataset.csv]")
        exit()
    dataset = "datasets/dataset_train.csv"
    if argc == 2:
        dataset = sys.argv[1]
    try:
        df = pd.read_csv(dataset)
        non_number = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
        df.drop(columns=non_number, inplace=True)
        show_histograms(df, "Hogwarts House")
        print("An homogeneous feature should have it's histograms stacked on top of each other.")
        print(
            "> Arithmancy and Care of Magical Creatures have an homogeneous score distribution between all four houses."
        )
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
