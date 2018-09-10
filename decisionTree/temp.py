def convertCharge(fname):
    f = open(fname, 'r')
    s = ''
    for line in f:
        tokens = line.split(",")
        if tokens[6] == 'anion': tokens[6] = 0
        elif tokens[6] == 'cation': tokens[6] = 1
        tokens = [str(x) for x in tokens]
        s += ','.join(tokens) + '\n'
    g = open("newfile", 'w')
    g.write(s)

convertCharge('oat1oct1.csv')
convertCharge('validation')
convertCharge('testdata')
