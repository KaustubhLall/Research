import random

def shuffleFile(curfile, newfile):
    '''
    Takes in a file and randomly shuffles all the samples in the file.

    '''
    lines = []

    f1 = open(curfile)

    for line in f1:
        lines.append(line)
    random.shuffle(lines)

    f2 = open(newfile, 'w')
    string = '\n'.join(lines)
    f2.write(string)
    f2.close()
    print("successfulyl shuffled the data points in %s and wrote to %s" % (curfile, newfile))


def splitData(datafile):
    '''
    Splits data into three files -- whose names are specified by the variables given below.
    Splits are defaulted to 80%, 10% and 10%, but may be manually set.
    '''

    designMatrixName = 'datafile'
    testDataName = 'testdata'
    validationDataName = 'validation'
    # design matrix, test matrix, validation matrix
    splits = [80, 10, 10]

    lines = []

    f = open(datafile)
    for line in f:
        lines.append(line)

    total = len(lines)

    designLen = int(splits[0] * total * 1e-2)     
    testLen, validationLen = int(1e-2 * splits[1] * total), int(1e-2 * splits[2] * total)
    print(len(lines), designLen, testLen, validationLen)

    f = open(designMatrixName, 'w')
    counter = 0
    for i in range(designLen):
        f.write(lines[i] + '\n')
    counter = designLen

    f = open(testDataName, 'w')
    for i in range(counter, counter + testLen):
        f.write(lines[i] + '\n')

    f = open(validationDataName, 'w')
    counter += testLen 
    for i in range(counter, counter + validationLen):
        f.write(lines[i] + '\n')

    print('Successfully divided %s into %s %s %s' % (datafile, designMatrixName, testDataName, validationDataName))
        
shuffleFile('newfile', 'new')
splitData('new')
    


