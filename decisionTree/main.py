from  util import *
from io import *

global root 
root = node()

# parse data file
S, L = createDataMatrix('newfile') 
print("Extracted Data from CSV, have %d features and %d points in the feature space" % (len(S[0]), len(S)))

def build_tree(n, S, L):
    '''
    Recursively build a ID3Tree.
    :params:
        n --> node being handled. The recursion starts at the root.
        S --> dataset associated with node n.
        L --> labels associated with dataset S.
    
    This is a recursive function. It will call itself on the children node of n,
    unless n is a "pure" node such that ever label in L is the same. A "pure"
    node is called a leaf node, and when reached predicts the label contained in
    L. 

    :return_value:
        n --> reference to the node at which the recursion terminates. 
    '''
    # isPure is a function that checks if L is comprised of only one unique
    # label.
    pure = isPure(L)

    # assign dataset and labels to the node n.
    n.S = S
    n.L = L

    # if the node is pure, we have reached the base case of our recursion. We
    # can return out of the function by returning a reference to the node object
    # n for other functions to use.
    if pure:
        n.leaf = True
        n.label = L[0]
        return n

    else:
        # find split. f --> feature to split over, t --> threshold to split at.
        f, t = fsr(S, L)
        # if f is a majority split, ie the node has only one label, end
        # recursion and set that node to be a leaf node (a prediction node).
        if f == 'majority':
            n.leaf = True
            n.label = t
            return n
        #print(f,t )
        # bugcheck to make sure a correct feature is returned
        assert f > -1

        # update node parameters otherwise and set the majority label, later to
        # be used for pruning.
        n.f = f
        n.t = t
        n.majlabel = major_label(L)
        
        # split the data for the yes and no branches of the decision tree with
        # corresponding labels. S --> dataset associated with the node. Sy -->
        # dataset for yes branch Sn --> dataset for no branch. L represents the
        # labels associated with this data.
        Sy = []
        Sn = []
        Ly = []
        Ln = []

        # v represents a vector of (feature1, feature2, ...) in our data matrix
        # S.
        for v in S:
            # if the f-th feature of v is leq the threshold, set is to the "yes"
            # branch of the node, as well as the corresponding label.
            if v[f] <= t:
                Sy.append(v)
                Ly.append(L[S.index(v)])
            else:
            # else the f-th feature corresponds to the "no" branch of the tree.
                Sn.append(v)
                Ln.append(L[S.index(v)])

        # assign children by recursively calling the same function on split
        # datset with the yes and no child respectively.
        n.ychild = build_tree(node(), Sy, Ly)
        n.nchild = build_tree(node(), Sn, Ln)
        return n
    
###### begin building the tree.

print("Building tree...", end='\r')
build_tree(root, S, L)
header("Building tree... Done!")

# print first three layers
header('L1/x: root')
print(root)
header('L2/1 : root->yes')
print(root.ychild)
header('L2/2 : root->no')
print(root.nchild)
header('L3/1 : root->yes->yes')
print(root.ychild.ychild)
header('L3/2 : root->yes->no')
print(root.ychild.nchild)
header('L3/1 : root->no->yes')
print(root.nchild.ychild)
header('L3/2 : root->no->no')
print(root.nchild.nchild)

# do training and test error
# for errors, we iterate over the training, test and validation dataset. The
# variable corr tracks all the correct predictions, and the variable i tracks
# the number of predictions made. Thus the error for any dataset is given by
# corr/i.

print("Testing Training Data...", end='\r')
corr = i = 0

# for every row of data in our dataset S, get the prediction from the tree and
# increment corr counter if the prediction matches the label.
for tensor in S:
    pred = prediction(root, tensor)
    if pred == L[i]:
        corr += 1
    i += 1
header("Testing Training Data... Done!")
print("Training Accuracy: %3.4f" % (corr/i))

# VS, Vl --> Test data and test labels resprectively.
VS, VL = createDataMatrix('testdata')
print("Testing Test Data...", end='\r')

# reset the corr counter for test error calculation.
corr = 0
i = 0
# repeat prediction process and check for the test data.
for tensor in VS:
    pred = prediction(root, tensor)
    if pred == VL[i]:
        corr += 1
    i += 1

header("Testing Test Data... Done!")
print("Test Accuracy: %3.4f" % (corr/i))

# prune the tree
header("Pruning Tree")

# Use a breadth-first search to traverse the tree, pruning from root, to left
# child to right child and so forth. The rule for pruning uses a hold-out
# validation technique. The validation error of the prediction made by the
# subtree is compared to the accuracy of the major label of the node. If the
# major label promises better performance, the subtree is pruned and the current
# node is converted to a leaf node that predicts the majority label. 
# Defns: majority label is the label that occurs most frequently in the dataset
# associated with the current node of the decision tree. Ex maj label of 
# 1, 0, 0, 0, 1 --> 0 but maj label of 0, 0, 1, 1 --> 1 or 0, at random.
from queue import Queue
q = Queue()
q.put(root)

suc = 0
while not q.empty():
    n = q.get()
    # check if we should replace it. v0, v1 are validation errors before and
    # after pruning and t0 and t1 represent test errors before and after
    # pruning.
    # suc counts the number of times puning happened.
    v0, v1, t0, t1 = replace(root, n, VS, VL)
    if v1 <= v0:
        suc += 1
        n.pruned = True
        print("%d round, validation err: %f, test err: %f" %(suc, v1, t1))
    else:
        # if children exist, push them to be processed next
        l = n.ychild
        r = n.nchild
        if l != None: q.put(l)
        if r != None: q.put(r)

# after pruning, check test-error
print("Testing Test Data after pruning...", end='\r')

# reset the corr counter for test error calculation.
corr = 0
i = 0
# repeat prediction process and check for the test data.
for tensor in VS:
    pred = prediction(root, tensor)
    if pred == VL[i]:
        corr += 1
    i += 1

header("Testing Test Data... Done!")
print("Test Accuracy: %3.4f" % (corr/i))

# print first three layers
header('L1/x: root')
print(root)
header('L2/1 : root->yes')
print(root.ychild)
header('L2/2 : root->no')
print(root.nchild)
header('L3/1 : root->yes->yes')
print(root.ychild.ychild)
header('L3/2 : root->yes->no')
print(root.ychild.nchild)
header('L3/1 : root->no->yes')
print(root.nchild.ychild)
header('L3/2 : root->no->no')
print(root.nchild.nchild)
while True: translateFeatures()
header("Program finished execution.")
