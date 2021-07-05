import matplotlib.pyplot as plt
import sys
import pandas as pd


def show_histograms(df: pd.DataFrame, features: list) -> None:
    rowlength = int(len(df.columns) / 2)
    fig, axs = plt.subplots(nrows=2, ncols=rowlength, figsize=(16, 8))
    all_axs = axs.flatten()
    group = df.groupby(["Hogwarts House"])
    i = 0
    for feature, values in df.items():
        if feature == "Hogwarts House" or (features and feature in features):
            continue
        group[feature].hist(alpha=0.5, ax=all_axs[i], legend=True)
        all_axs[i].set_title(feature)
        i += 1
    while i < len(all_axs):
        all_axs[i].axis("off")
        i += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    print("An homogeneous feature should have it's histograms stacked on top of each other.")
    print("> Arithmancy and Care of Magical Creatures have an homogeneous score distribution between all four houses.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: histogram.py [dataset.csv]")
        exit()
    dataset = sys.argv[1]
    features = []
    try:
        df = pd.read_csv(dataset)
        df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"], inplace=True)
        show_histograms(df, features)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
    except pd.errors.ParserError as err:
        print("Invalid dataset dataset: {}".format(err))
