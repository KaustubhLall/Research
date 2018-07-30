from collections import Counter
from operator import itemgetter
from math import log
from random import choice

def fsr(S, l, num=22):
    '''
    Given a dataset, return (f, t) which results in the maximum IG.
    '''
    num = len(S[0])
    assert len(S[0]) == num
    features = list(range(num)) 
    # calculate and store information gain for each feature as (f, t)
    gain = []

    for f in features:
        # calculate the (maximum of) n-1 possible split values
        splits = []
        # stores UNIQUE coordinates of points along the f-axis
        feature_col = []
        # information gain array for this particular feature. Stores (f, t)
        gain_f = []

        # extract feature column from data matrix
        for i in range(len(S)):
            val = S[i][f]
            if val not in feature_col:
                feature_col.append(val)

        # sort the values of the features
        feature_col = sorted(feature_col)

        # if all values of the feature are the same
        if(len(feature_col) <= 1): 
            gain.append((0, feature_col[0]))
            continue

        for i in range(len(feature_col) - 1):
            t = (feature_col[i+1] + feature_col[i]) / 2 
            splits.append(t) 
        # now that we have the splits for the feature, we will find information
        # gain for that particular (f, t) pair. 
        for t0 in splits:
            # params: feature, threshold, vector of features
            # stores (ig, threshold)
            gain_f.append((IG(t0, [x[f] for x in S], l), t0))

        # select the higest information gain for that feature
        (ig, t0) = max(gain_f, key=itemgetter(0))
        gain.append((ig, t0))

    # return the feature, threshold
    best_ig = max(gain, key=itemgetter(0))
    return gain.index(best_ig), best_ig[1]

def IG(t, v, l):
    '''
    Given a feature vector, find its information gain. Ie - find H(X) - H(X|Z).
    '''
    # first, we define our random variable. It is a dict which stores P(X=0) and
    # P(X=1)
    X = Counter(l)
    X[0] = X[0] / (X[0] + X[1])
    X[1] = 1 - X[0]
    # H(X) = p1 log(p1) + p2 log(p2)
    entropy = H(X)
    conditional_entropy = HcondZ(t, v, l)
    return entropy - conditional_entropy


def H(X):
    '''
    Find the entropy of a random variable X.
    '''
    if X[0] == 1 or X[1] == 1: return 0
    return sum(-p[1] * log(p[1]) for p in X.items())

def HcondZ(t, v, l):
    '''
    Find the conditional entropy of X given Z.
    H(X|Z) = sum over all z in Z: H(X|Z=z)
    '''
    # label = (0, 1)
    y = [0, 0]
    n = [0, 0]
    for i in range(len(v)):
        val = v[i]
        label = l[i]

        if val <= t:
            y[label] += 1
        else:
            n[label] += 1

    XcondZy = {}
    XcondZn = {}

    # random variable X|Z=yes
    XcondZy[0] = y[0] / sum(y)
    XcondZy[1] = 1 - XcondZy[0]

    # random variable X|Z=no
    XcondZn[0] = n[0] / sum(n)
    XcondZn[1] = 1 - XcondZn[0]

    # Pr(Z=yes), Pr(Z=no)
    pZy = sum(y) / (sum(n) + sum(y))
    pZn = sum(n) / (sum(n) + sum(y))

    HcondZy = H(XcondZy)
    HcondZn = H(XcondZn)
    return pZy * HcondZy + pZn * HcondZn 

def parse_input(s, n=22):
    '''
    Takes a given filename with n features and one label.
    Returns datamatrix, labels.
    '''
    d = []
    labels = []

    f = open(s)
    i = 0
    for line in f:
        features = line.split()
        label = int(float(features[n]))
        features = [float(x) for x in features[:n]]
        d.append(features)
        labels.append(label)
        i += 1
    return d, labels

def isPure(v):
    '''
    Check if there is only a unique label.
    '''
    if type(v[0]) != type(0): print(v)
    return len(set(v)) == 1

def major_label(l):
    c = Counter(l)

    if c[0] == c[1]:
        # if majority labels occus uniformly
        print("Warning both labels are equiprobable (Called major_label)")
        return choice([0, 1])

    return 0 if c[0] > c[1] else 1

class node():
    # associated data
    S = []
    # associated labels
    L = []
    # feature index
    f = -1
    # threshold
    t = -1
    # is the current node a leaf node ie impure?
    leaf = False
    # label
    label = -1
    # pointers to children
    ychild = nchild = None
    # majority label
    majlabel = -1
    # check if tree pruned
    pruned = False

    def decision(self, v):
        '''
        For a given vector v, what does the splitting rule say?
        Return value is either a pointer to a node, or the label.
        Return Value:
        if leaf node, returns the label
        if not, returns a pointer to correct child. 
        Check return value using type(rval) == node.
        '''
        if self.leaf:
            return self.label
        if self.pruned:
            return self.majlabel
        return  self.ychild if v[self.f] <= self.t else self.nchild

    def __repr__(self):
        '''
        For a given node, instructions to print it and any relevant details.
        '''
        s = '-' * 14
        s += '\n'
        if self.label == -1: 
           s += "|{:^12s}|".format("f = " + str(self.f))
           s += '\n'
           s += "|{:^12s}|".format("t = " + str(self.t))
           s += '\n'
        else: 
           s += "{:^12s}".format("Label: " + str(self.label))
           s += "\n"
        # s += "Majority Label: " + str(self.majlabel)
        s += "|{:^12s}|".format('|S| : %04d' % len(self.S))
        s += '\n'
        s += "|{:^12d}|".format(self.pruned)
        s += '\n'
        s += '-' * 14
        return s

def prediction(n, v):
    '''
    Takes a node and outputs a single label.
    '''
    cur = n

    while type(cur) != int:
        cur = cur.decision(v)
    return cur

def header(s, len=120):
    print("-"*len)
    print('{:^120s}'.format(s))
    print("-"*len)

def replace(root, n, SV, SL):
    '''
    Classifies validation set on given node, 
    '''
    pre_err = 0
    err = 0
    i = 0

    S, L = parse_input('pa2validation.txt')
    assert len(S[0]) == 22
    for v in S:
        pred = prediction(root, v)
        if pred != L[i]:
            err += 1
        i += 1
    pre_err = err/i

    err = 0
    i = 0
    n.pruned = True
    for v in S:
        pred = prediction(root, v)
        if pred != L[i]:
            err += 1
        i += 1
    n.pruned = False
    post_err = err/i

    ## training error ##
    err = 0
    i = 0

    for v in SV:
        pred = prediction(root, v)
        if pred != SL[i]:
            err += 1
        i += 1
    t_err0 = err/i


    err = 0
    i = 0

    n.pruned = True

    for v in SV:
        pred = prediction(root, v)
        if pred != SL[i]:
            err += 1
        i += 1
    t_err1 = err/ i

    n.pruned = False
    return pre_err, post_err, t_err0, t_err1

def createDataMatrix(fname, cols=[], ignoreHeader=True):
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
    header = False
    for line in  f:
        if header:
            tokens = line.split(',')[8:]

            # extract the label
            # 0 - oat1
            # 1 - oct1
            label = tokens[-1].strip().lower()
            if label == 'oat1': labels.append(0)
            elif label == '1-oct': labels.append(1)
            else: 
                print(label)
                raise AssertionError

            # clean up the tokens
            tokens = [float(x.strip()) for x in tokens[:-1]]


            # extract the vector into a data matrix
            # check if any columns were specified
            vector = []
            if len(cols) == 0:
                dm.append(tokens[:-1])
            else:
                vector = [tokens[i] for i in cols]
                dm.append(vector)
        else: header = True
    return dm, labels




