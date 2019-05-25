def read_data(data):
    import pandas
    dataframe = pandas.read_csv(data,header=None)
    return dataframe.values