import os
import sys
import math
import pandas as pd


def percentile(serie: list, percentile: int) -> float:
    """
    Closest ranks with linear interpolation (C = 1)
    https://en.wikipedia.org/wiki/Percentile
    """
    count = len(serie)
    x = (percentile / 100) * (count - 1)
    x_floor = math.floor(x)
    value = x_floor if x_floor >= 0 else 0
    value_next = x_floor + 1 if (x_floor + 1) < count - 1 else count - 1
    in_serie = serie[value]
    next_in_serie = serie[value_next]
    frac = x - x_floor
    return in_serie + frac * (next_in_serie - in_serie)


def describe_serie(serie: pd.Series) -> dict:
    """
    Calculate the number of elements, average, standard deviation, min,
    25, 50 and 75 percentiles and the max of a given serie, and returns them in a dict.
    NaN and empty elements are ignored.
    """
    cleaned = serie.dropna().sort_values().to_list()
    count = len(cleaned)
    if count == 0:
        return {}
    min = cleaned[0]
    max = cleaned[-1]
    s_sum = sum(row for row in cleaned)
    mean = s_sum / count
    var = sum((row - mean) ** 2 for row in cleaned) / (count)
    std = math.sqrt(var)
    per25 = percentile(cleaned, 25)
    per50 = percentile(cleaned, 50)
    per75 = percentile(cleaned, 75)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min,
        "per25": per25,
        "per50": per50,
        "per75": per75,
        "max": max,
    }


def formatted_description(name: str, serie: pd.Series) -> dict:
    """
    Format the output of the describe_serie function and add the name of the feature and longest string as the length.
    """
    raw_results = describe_serie(serie)
    results = {
        "name": name,
        "count": "{:.3f}".format(raw_results["count"]),
        "mean": "{:.3f}".format(raw_results["mean"]),
        "std": "{:.3f}".format(raw_results["std"]),
        "min": "{:.3f}".format(raw_results["min"]),
        "per25": "{:.3f}".format(raw_results["per25"]),
        "per50": "{:.3f}".format(raw_results["per50"]),
        "per75": "{:.3f}".format(raw_results["per75"]),
        "max": "{:.3f}".format(raw_results["max"]),
    }
    # Set length to the longest column + 2 for padding
    results["length"] = sorted(len(v) for v in results.values())[-1] + 2
    return results


def describe(series: pd.DataFrame, output: bool = True) -> list:
    described_series = []
    for key, values in series.items():
        described_series.append(formatted_description(key, values))
    if output:
        if len(described_series) == 0:
            print("No set of features available.")
        else:
            rows = {
                "name": "name",
                "count": "count",
                "mean": "mean",
                "std": "std",
                "min": "min",
                "25%": "per25",
                "50%": "per50",
                "75%": "per75",
                "max": "max",
            }
            # Calculate the set of features to show per line
            try:
                terminal_size = os.get_terminal_size().columns
            except:
                terminal_size = -1
            max_feature = len(described_series)
            sets = []
            current_set = []
            current_size = 7
            i = 0
            while i < max_feature:
                if current_size + described_series[i]["length"] >= terminal_size:
                    if len(current_set) == 0:
                        current_set.append(described_series[i])
                    sets.append(current_set)
                    current_set = []
                    current_size = 7
                else:
                    current_size += described_series[i]["length"]
                    current_set.append(described_series[i])
                i += 1
            if current_set:
                sets.append(current_set)
            # Display all found set per line
            max_set = len(sets)
            for index, feature_set in enumerate(sets):
                for row, key in rows.items():
                    if row == "name":
                        print("       ", end="")
                    else:
                        print("{0:<7}".format(row), end="")
                    for description in feature_set:
                        print("{0:>{length}}".format(description[key], length=description["length"]), end="")
                    print("")
                if index < max_set - 1:
                    print("")
    return described_series


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: describe.py [dataset.csv]")
        exit()
    dataset = sys.argv[1]
    try:
        df = pd.read_csv(dataset)
        df.drop(columns=["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"], inplace=True)
        describe(df)
    except IOError as err:
        print("Failed to read dataset: {}".format(err))
    except pd.errors.ParserError as err:
        print("Invalid dataset: {}".format(err))
