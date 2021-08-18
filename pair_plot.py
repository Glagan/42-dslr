import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "Ravenclaw": "indianred",
    "Slytherin": "green",
    "Gryffindor": "gold",
    "Hufflepuff": "teal",
}


def show_pair_plot(df: pd.DataFrame, class_index: str):
    # df.set_index("Index", drop=True, inplace=True)
    length = len(df.columns) - 1
    last_row = length * (length - 1)
    grouped = df.groupby(class_index)
    fig, dim_axs = plt.subplots(nrows=length, ncols=length, figsize=(16, 8), gridspec_kw={"wspace": 0, "hspace": 0})
    axs = dim_axs.flatten()
    i = 0
    for feature, values in df.items():
        if feature == class_index:
            continue
        for against_feature, values in df.items():
            if against_feature == class_index:
                continue
            # Remove ticks and labels
            if (i % length) == 0:
                axs[i].set_ylabel("{}{}".format(feature, "\n" if (i % 2) == 1 else ""))
            if i >= last_row:
                axs[i].set_xlabel("{}{}".format("\n" if (i % 2) == 1 else "", against_feature))
            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_xticks([])
            axs[i].grid(False)
            # Display each features against each others
            if feature == against_feature:
                grouped[feature].hist(alpha=0.5, ax=axs[i])
            else:
                for house, y in grouped:
                    axs[i].scatter(x=y[against_feature], y=y[feature], marker=".", c=colors[house], alpha=0.5)
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
        non_number = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
        df.drop(columns=non_number, inplace=True)
        print("Generating plot...")
        show_pair_plot(df, "Hogwarts House")
        print(
            "The diagonal of the scatter matrix is the dispersion of the feature values each rows and columns show the feature plotted against each other features (X/Y reversed on each side of the diagonal)."
        )
        print(
            "Features that extract 1 house from all of the others should be selected since the logistic regression calculate weights for the houses one by one."
        )
        print(
            "> *Arithmancy*, *Potions* and *Care of Magical Creatures* should be avoided since their dispersion between all 4 houses is homogeneous."
        )
        print(
            "> *Divination*, *Muggle Studies*, *History of Magic*, *Transfiguration*, *Charms* and *Flying* should be good features to use in the logistic regression."
        )
        print(
            "> The remaining features can be used as they provide different set of houses against other houses and are complementary."
        )
    except IOError as err:
        print(f"Failed to read dataset: {err}")
    except pd.errors.ParserError as err:
        print(f"Invalid dataset: {err}")
