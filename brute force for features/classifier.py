from sklearn import tree
from datacontainer import *
from gen_combinations import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve
def testDecisionTree(trainx, trainy, testx, testy, cols):
    '''
    Tests a decision tree on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param dc: the main datacontainer to use
    :param cols: the chosen cols to use for the test
    :param labelind: index of label column
    :return: error, cols as a string, auc, roc
    '''

    print('fitting classifier DT')
    # fit tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainx, trainy)
    print('done fitting!')

    # error check
    corr = 0
    total = 0

    pred = clf.predict(testx)
    print(list(pred))

    for p in pred:
        if p == testy[total]: corr += 1
        total += 1

    # do auc and roc here
    auc = roc_auc_score(testy, pred)
    #roc = roc_curve(testy, pred)
    return corr/total, ''.join(["%02d" % x for x in cols]), auc#, roc


def testRFW(trainx, trainy, testx, testy, cols):
    '''
    Tests a random forest walk classifier on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param dc: the main datacontainer to use
    :param cols: the chosen cols to use for the test
    :param labelind: index of label column
    :return: error, cols as a string
    '''

    print('fitting classifier RWF')
    # train classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(trainx, trainy)
    print('done fitting!')
    # error check
    corr = 0
    total = 0
    pred = list(clf.predict(testx))
    print(pred)
    for p in pred:
        if p == testy[total]: corr += 1
        total += 1
    auc = roc_auc_score(testy, pred)
    #roc = roc_curve(testy, pred)
    return corr/total, ''.join(["%02d" % x for x in cols]), auc#, roc




