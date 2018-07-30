def createDataMatrix(fname, cols=[]):
    '''
    Returns a data matrix from a .csv file.
    :params:
        fname   --> name/path of file to open.
        cols    --> which columns to scrape. Default is all.
    :return:
        data matrix, labels as a tuple
    '''
    f = open(fname)
    dm = []
    labels = []
    for line in  f:
        tokens = line.split(',')

        # clean up the tokens
        tokens = [x.strip() for x in tokens]

        # extract the label
        labels.append(tokens[-1])

        # extract the vector into a data matrix
        # check if any columns were specified
        vector = []
        if cols == []:
            dm.append(labels[:-1])
        else:
            vector = [tokens[i] for i in cols]
            dm.append(vector)
    return dm, labels




