import csv


def read_dataset(dataset: str):
    """
    Open a CSV dataset and assign each rows to a list in the dict series[header].
    Return False on Exception and the series dict on success.
    """
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
        return False
    except Exception as err:
        print('Unknown error: {}.'.format(err))
        return False
    return series
