from datacontainer import *


def generateSequences(n, k):
    '''
    Generates all possible combinations of integer sequences of length k using the first 0, ..., n integers.
    :param n: size of chosing pool.
    :param k: size of each pool
    :return: list of all possible sequences.
    '''
    return list(choose_iter(list(range(n)), k))

def choose_iter(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i + 1:len(elements)], length - 1):
                yield (elements[i],) + next
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next


