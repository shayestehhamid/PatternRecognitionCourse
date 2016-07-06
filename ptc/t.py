from glob import glob
import numpy as np
import scipy as sc


class Graph():
    
    def __init__(self, n):
        self.nodes = ['ff' for x in xrange(n)]
        self.matrix = [[0 for i in xrange(n)] for x in xrange(n)]


files = glob('./corrected_structures/*')
cn = 0
for data in files:
    print cn
    cn += 1

    f = open(data)
    print data

    next(f);next(f);next(f); 
    V, E = [int(x) for x in (next(f).split()[:2])]
    print V, E
    g = Graph(V)
    for v in range(V):
        c = next(f).split()[3]
        g.nodes[v] = c


