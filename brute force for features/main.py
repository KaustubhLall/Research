import datetime
from time import time
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
from sklearn.metrics import roc_auc_score


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
    testx = subsetDataContainer(testx, list(range(6, testx.numcols)))
    trainx = subsetDataContainer(trainx, list(range(6, trainx.numcols)))

    # generate all feature sequences
    print('generating sequences...')
    seq = generateSequences(testx.numcols, k)
    print('generated %d sequences' % len(seq))
    totalseq = len(seq)

    f = open(results, 'w')

    prog = 0
    bestDT = 0, ''
    bestRFW = 0, ''
    bestLR = 0, ''
    bestNB = 0, ''
    bestKNN = 0, ''

    # run the sequences via a decision tree and then repeat for RFW
    print("starting test..., %d choose %d sequences" % (k, trainx.numcols))
    # f.write(','.join(testx.header))
    timeStart = time()
    for pattern in seq:
        subtrainx = subsetDataContainer(trainx, pattern)
        subtestx = subsetDataContainer(testx, pattern)

        subtrainx = subtrainx.dataMatrix
        subtestx = subtestx.dataMatrix

        resDT = testDecisionTree(subtrainx, trainy, subtestx, testy, pattern)
        resRFW = testRFW(subtrainx, trainy, subtestx, testy, pattern)
        resLR = testLogisticRegressor(subtrainx, trainy, subtestx, testy, pattern)
        resKNN = testKNN(subtrainx, trainy, subtestx, testy, pattern)
        resNB = testNaiveBayes(subtrainx, trainy, subtestx, testy, pattern)

        # do auc and roc comparisons here

        if resDT[0] > bestDT[0]:
            bestDT = resDT

        if resRFW[0] > bestRFW[0]:
            bestRFW = resRFW

        if resKNN[0] > bestKNN[0]:
            bestKNN = resKNN

        if resLR[0] > bestLR[0]:
            bestLR = resLR

        if resNB[0] > bestNB[0]:
            bestNB = resNB

        prog += 1
        featuresDT = []
        featuresRFW = []
        featuresNB = []
        featuresLR = []
        featuresKNN = []

        for i in range(0, len(resDT[1]) - 1, 2):
            featuresDT.append(trainx.header[int(resDT[1][i:i + 2])])
            featuresRFW.append(trainx.header[int(resRFW[1][i:i + 2])])
            featuresNB.append(trainx.header[int(resNB[1][i:i + 2])])
            featuresLR.append(trainx.header[int(resLR[1][i:i + 2])])
            featuresKNN.append(trainx.header[int(resKNN[1][i:i + 2])])

        # write test results to file
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (resDT[0], resRFW[0], resNB[0], resLR[0], resKNN[0], ','.join(featuresDT)))
        curtime = time()
        predtime = ((curtime - timeStart) * (totalseq - prog) / prog)
        predtime = str(datetime.timedelta(seconds=predtime)).split('.')[0]
        print("Classified %d/%d %03.02f%% \t current bests: %02.02f, %02.02f, %02.02f, %02.02f, %02.02f, ETA: %s" % (
            prog, totalseq, prog / totalseq * 1e2, bestDT[0], bestRFW[0], bestNB[0], bestKNN[0], bestLR[0], predtime),
              end='\r')

    f.close()
    print("Finished!")


def enumerate_over_features(ftest, ftrain, f=[2, 5, 6, 7, 8, 9]):
    for e in f:
        print("Running %s" % e)
        runall(ftest, ftrain, 'results' + str(e), e)


enumerate_over_features('test.csv', 'train.csv')
