import os
import sys
import math
import csv


def percentile(serie: list, percentile: int) -> float:
    """
    Closest ranks with linear interpolation (C = 1)
    https://en.wikipedia.org/wiki/Percentile
    """
    count = len(serie)
    x = (percentile / 100) * (count - 1)
    x_floor = math.floor(x)
    value = x_floor if x_floor >= 0 else 0
    value_next = x_floor + \
        1 if (x_floor + 1) < count - 1 else count - 1
    in_serie = serie[value]
    next_in_serie = serie[value_next]
    frac = x - x_floor
    return in_serie + frac * (next_in_serie - in_serie)


def describe_column(name: str, serie: list) -> dict:
    """
    Calculate the number of elements, average, standard deviation, min, 25, 50 and 75 percentiles and the max of a given serie,
    and returns them in a dict.
    NaN and empty elements are ignored.
    """
    cleaned = sorted(float(row) for row in serie if row != '' and row != None)
    count = len(cleaned)
    if count == 0:
        return ()
    min = cleaned[0]
    max = cleaned[-1]
    s_sum = sum(row for row in cleaned)
    mean = s_sum / count
    var = sum((row - mean) ** 2 for row in cleaned) / (count)
    std = math.sqrt(var)
    per25 = percentile(cleaned, 25)
    per50 = percentile(cleaned, 50)
    per75 = percentile(cleaned, 75)
    result = {
        'name': name,
        'count': '{:.3f}'.format(count),
        'mean': '{:.3f}'.format(mean),
        'std': '{:.3f}'.format(std),
        'min': '{:.3f}'.format(min),
        'per25': '{:.3f}'.format(per25),
        'per50': '{:.3f}'.format(per50),
        'per75': '{:.3f}'.format(per75),
        'max': '{:.3f}'.format(max)}
    # Set length to the longest column + 2 for padding
    result['length'] = sorted(len(v) for v in result.values())[-1] + 2
    return result


def describe(series: dict) -> None:
    described_series = []
    for key, values in sorted(series.items()):
        # Check if the list only contains numerical or empty values
        is_number = True
        for v in values:
            try:
                if v != None and v != '':
                    float(v)
            except:
                is_number = False
                break
        if is_number:
            described_series.append(describe_column(key, values))
    if len(described_series) == 0:
        print('No set of features available.')
    else:
        rows = {'name': 'name', 'count': 'count', 'mean': 'mean', 'std': 'std',
                'min': 'min', '25%': 'per25', '50%': 'per50', '75%': 'per75', 'max': 'max'}
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
            if current_size + described_series[i]['length'] >= terminal_size:
                if len(current_set) == 0:
                    current_set.append(described_series[i])
                sets.append(current_set)
                current_set = []
                current_size = 7
            else:
                current_size += described_series[i]['length']
                current_set.append(described_series[i])
            i += 1
        if current_set:
            sets.append(current_set)
        # Display all found set per line
        max_set = len(sets)
        for index, feature_set in enumerate(sets):
            for row, key in rows.items():
                if row == 'name':
                    print('       ', end='')
                else:
                    print('{0:<7}'.format(row), end='')
                for description in feature_set:
                    print('{0:>{length}}'.format(
                        description[key], length=description['length']), end='')
                print('')
            if index < max_set - 1:
                print('')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: describe.py [dataset.csv]')
        exit()
    dataset = sys.argv[1]
    series = {}
    try:
        with open(dataset) as csvDataFile:
            csvReader = csv.DictReader(csvDataFile)
            for row in csvReader:
                # Separate each columns in a row to a distinct list
                for key in row.keys():
                    series.setdefault(key, [])
                    series[key].append(row.get(key))
    except IOError as err:
        print('Failed to read dataset: {}.'.format(err))
    except Exception as err:
        print('Unknown error: {}.'.format(err))
    describe(series)
