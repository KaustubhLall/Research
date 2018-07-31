from  util import *
from io import *

global root 
root = node()

# parse data file
S, L = createDataMatrix('oat1oct1.csv') 
print("Extracted Data from CSV, have %d features and %d points in the feature space" % (len(S[0]), len(S)))

def build_tree(n, S, L):
    '''
    Recursively build a ID3Tree.
    '''
    pure = isPure(L)
    n.S = S
    n.L = L

    if pure:
        n.leaf = True
        n.label = L[0]
        return n

    else:
        # find split
        f, t = fsr(S, L)
        if f == 'majority':
            n.leaf = True
            n.label = t
            return n
        #print(f,t )
        assert f > -1

        # update node parameters
        n.f = f
        n.t = t
        n.majlabel = major_label(L)
        
        # split the data
        Sy = []
        Sn = []
        Ly = []
        Ln = []

        for v in S:
            if v[f] <= t:
                Sy.append(v)
                Ly.append(L[S.index(v)])
            else:
                Sn.append(v)
                Ln.append(L[S.index(v)])

        # assign children 
        n.ychild = build_tree(node(), Sy, Ly)
        n.nchild = build_tree(node(), Sn, Ln)
        return n
    
###### begin tree

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


print("Testing Training Data...", end='\r')
corr = i = 0
for tensor in S:
    pred = prediction(root, tensor)
    if pred == L[i]:
        corr += 1
    i += 1
header("Testing Training Data... Done!")
print("Training Accuracy: %3.4f" % (corr/i))


VS, VL = createDataMatrix('testdata')
print("Testing Test Data...", end='\r')

corr = 0
i = 0
for tensor in VS:
    pred = prediction(root, tensor)
    if pred == VL[i]:
        corr += 1
    i += 1

header("Testing Test Data... Done!")
print("Test Accuracy: %3.4f" % (corr/i))

# prune the tree
header("Pruning Tree")

# go as bfs
from queue import Queue
q = Queue()
q.put(root)

suc = 0
while not q.empty():
    n = q.get()
    # check if we should replace it
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
