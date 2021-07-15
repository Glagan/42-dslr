import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "Ravenclaw": "indianred",
    "Slytherin": "green",
    "Gryffindor": "gold",
    "Hufflepuff": "teal",
}


def show_pair_plot(df: pd.DataFrame):
    # df.set_index("Index", drop=True, inplace=True)
    length = len(df.columns) - 1
    last_row = length * (length - 1)
    grouped = df.groupby("Hogwarts House")
    fig, dim_axs = plt.subplots(nrows=length, ncols=length, figsize=(16, 8), gridspec_kw={"wspace": 0, "hspace": 0})
    axs = dim_axs.flatten()
    i = 0
    for feature, values in df.items():
        if feature == "Hogwarts House":
            continue
        for against_feature, values in df.items():
            if against_feature == "Hogwarts House":
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
    print(
        "The diagonal of the scatter matrix is the dispersion of the feature values each rows and columns show the feature plotted against each other features (X/Y reversed on each side of the diagonal)."
    )
    print(
        "Features that have 4 visible groups against at least multiple other features should be selected since they will make the logistic regression much easier and accurate."
    )
    print(
        "> *Arithmancy*, *Potions* and *Care of Magical Creatures* should be avoided since they all have only two visible groups against all other features."
    )
    print(
        "> *Astronomy*, *Herbology*, *Defense Against the Dark Arts*, *Ancient Runes* and *Charms* should be good features to use in the logistic regression."
    )
    print(
        "> *Astronomy* and *Herbology* are also complementary, their dispersion (on the diagonal) shows that they have a different dispersion for two different groups of two."
    )
    print(
        "> *Divination*, *Muggle Studies*, *History of Magic*, *Transfiguration* and *Flying* could help the logistic regression but also might introduce false positive since they only have 3 visible groups."
    )
    print(
        '> Like with *Astronomy* and *Herbology*, the three features *Divination*, *Muggle Studies* and *History of Magic* are "complementary", they each extract one group from the other 3 groups.'
    )
    print('> They also extract features that have a low dispersion in the "good" features selected.')


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
        show_pair_plot(df)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
