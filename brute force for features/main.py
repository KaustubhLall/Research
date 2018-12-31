# index where the class is
classind = 2
'''
This file will have a method to generate all possible n C k features of a given dataset.
The dataset in question has to be pre-processed and some columns may be dropped.

Todo:
    1. Preprocess given dataset - cols 0 1 4 5 have no useful data. We can drop these.
    2. After this, we have class in col 2, we can extract that as our label vector and drop the column.
    3. This leaves us with 20 columns, we want a ballpark 5 features. This is 20 C 5 * 3 files to be generated,
    which totals about 46,512 files. This may slow down disk reads/writes - we may just chose to store this in memory?
    4. Need to run a classifier on each of these features and remember the results. Once we have a new datacontainer,
    we can write that to csv and use that as our to-go datacontainer.


NOTES:
 To get a column by name from a datacontainer d, use d.header.index('name of column header') and pass that result
 into d.dropcol().
'''

from classifier import *
from datacontainer import *
from gen_combinations import *
from sklearn.metrics import  roc_auc_score

def preprocess(target, fname='oat1oat3preprocessed.csv'):
    df = DataContainer(fname)

    for e in [5, 4, 1, 0]:
        df.dropcol(e)

    classes = df.getcol(0)
    df.dropcol(0)

    df.writeCSV(target)

def runall(testname, trainname, results, k=5):
    # load test and training "main" datacontainers
    testx = DataContainer(testname)
    trainx = DataContainer(trainname)

    # extract labels and drop the columns
    testy = [x[0] for x in subsetDataContainer(testx, [classind]).dataMatrix]
    trainy = [x[0] for x in subsetDataContainer(trainx, [classind]).dataMatrix]

    # clean columns we dont need
    testx = subsetDataContainer(testx, [3] + list(range(5, testx.numcols)))
    trainx = subsetDataContainer(trainx, [3] + list(range(5, trainx.numcols)))

    # generate all feature sequences
    print('generating sequences...')
    seq = generateSequences(testx.numcols, k)
    print('generated %d sequences' % len(seq))
    totalseq = len(seq)

    f = open(results, 'w')

    prog = 0
    bestDT = 0, ''
    bestRWF = 0, ''
    # run the sequences via a decision tree and then repeat for RFW
    print("starting test...")
    for pattern in seq:
        print("Subsetting from main datacontainer")
        subtrainx = subsetDataContainer(trainx, pattern)
        subtestx = subsetDataContainer(testx, pattern)
        
        subtrainx = subtrainx.dataMatrix
        subtestx = subtestx.dataMatrix
        
        resDT = testDecisionTree(subtrainx, trainy, subtestx, testy, pattern)
        resRWF = testRWF(subtrainx, trainy, subtestx, testy, pattern)

        # do auc and roc comparisons here

        if resDT[0] > bestDT[0]:
            bestDT = resDT

        if resRWF[0] > bestRWF[0]:
            bestRWF = resRWF

        # write test results to file
        f.write("%s \t %s" % (resDT, resRWF))
        prog += 1
        print("Classified %d/%d\n current bests: %s \t %s" % (prog, totalseq, bestDT, bestRWF), end='\r')

    f.close()
    print("Finished!")


runall('test.csv','train.csv','results', 5)

