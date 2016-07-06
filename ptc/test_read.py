from glob import glob
import numpy as np
import scipy as sc
import sklearn

class Graph():
    
    def __init__(self, n):
        self.title = ""
        self.nodes = ['ff' for x in xrange(n)]
        self.matrix = [[0 for i in xrange(n)] for x in xrange(n)]
        self.res = {}
        self.deg = []
        self.N = len(self.nodes)
        
        self.distance_value = []
        
        
        
        self.features = []
        
        
        
    def __run__(self):
        
        
        for i in xrange(self.N):
            self.distance_value.append(self.distances(i)[1])
        self.compute_deg()
        
        self.features.append(self.ave_deg()) # f-1
        self.features.append(self.ave_clustering_coeffitient()) # f-2
        self.features.extend(self.ave_max_min_effective_eccentricity()) # f-3, f-4, f-5
        self.features.append(self.ave_path_len()) # f-6
        ####################### f-7 is left!
        self.features.append(self.central_point())
        self.features.append(self.giant_connected_ratio()) # f-8
        self.features.append(self.percent_isol_node()) # f-9
        self.features.append(self.percent_end_point()) # f-10
        self.features.append(self.node_no()) # f-11
        self.features.append(self.edge_no()) # f-12
        self.features.extend(self.eigenValue_features()) # f-13, f-14, f-15, f-16, f-17
        self.features.append(self.label_entropy()) # f-18
        self.features.append(self.ave_impurity_deg()) # f-19
        self.features.append(self.link_impurity()) # f-20
        
        
    
    
    def __str__(self):
        return self.title
    
    
    def compute_deg(self):
        for node in self.matrix:
            self.deg.append(sum([1 for x in node if x > 0]))
            
    
    def ave_deg(self): #f-1
        return sum(self.deg)/float(self.N)

    
    def neighbors(self, x): #helper function
        return [i for i in xrange(self.N) if self.matrix[x][i] > 0]
    
    
    def dfs(self, root, seen): #helper function
        seen.add(root)
        res = 1
        for x in self.neighbors(root):
            if x not in seen:
                res += self.dfs(x, seen)
        return res
                
    
    def count_cluster(self, x): # helper function
        cn = 0
        for nghb in self.neighbors(x):
            for n2 in self.neighbors(nghb):
                if n2 != x and self.matrix[n2][x]:
                    cn += 1
        return cn
    
    
    def ave_clustering_coeffitient(self): #f-2
        ss = 0.0
        for node in xrange(self.N):
            ss += ((self.count_cluster(node))/((self.deg[node]**2 - self.deg[node])/2)) if (self.deg[node] != 0 and self.deg[node] != 1) else 0
        return ss/self.N
    
    
    def distances(self, root): # helper function
        froot = root
        distance = [0 for i in xrange(self.N)]
        distance[root] = 0
        stack = [root]
        # append to add, pop to remove
        while len(stack):
            root = stack.pop()
            for nghb in self.neighbors(root):
                if not distance[nghb]:
                    distance[nghb] += distance[root] + 1
                    stack.append(nghb)
        
        
        return froot, distance
    
    
    def effective_eccentricity(self, root): # helper function ########### get to check!
        distance = sorted(self.distance_value[root], reverse=True)
        #print distance, root
        
        #print distance, distance[((self.N)/10)]
        
        return distance[(self.N)/10]
        
    
    
    def ave_max_min_effective_eccentricity(self): #f-3, f-4, f-5 ### has  to get check!!
        ecc = []
        for x in xrange(self.N):
            ecc.append(self.effective_eccentricity(x))
        
        return sum(ecc)/float(self.N) ,max(ecc), min(ecc)
    
    
    def ave_path_len(self): #f-6
        closeness = 0.0
        for x in xrange(self.N):
            dist = self.distance_value[x]
            closeness += (float(self.N-1)/(sum(dist)))
        return closeness/self.N
        
    
    def central_point(self): #f-7
        ds = []
        for x in self.distance_value:
            ds.extend(x)
        ds = sorted(ds, reverse=True)
        rad = ds[len(ds)/10]
        cn = 0
        for x in self.distance_value:
            if max(x) == rad:
                cn += 1
        return cn/float(self.N)
        
        
    def giant_connected_ratio(self): #f-8
        seen = set()
        size = 0
        for i in xrange(self.N):
            if i in seen:
                continue
            else:
                size = max(size, self.dfs(i, set()))
        return float(size)/self.N

    
    def percent_isol_node(self): #f-9
        return self.deg.count(0) / float(self.N)
    
    def percent_end_point(self): #f-10
        return self.deg.count(1) / float(self.N)

    def node_no(self): #f-11
        return self.N

    def edge_no(self): #f-12
        return sum(self.deg)/2
    
    
    def eigen_values(self): # helper function
        gg = np.array(self.matrix)
        for i in xrange(gg.shape[0]):
            for j in xrange(gg.shape[1]):
                gg[i, j] = 1 if gg[i, j] != 0 else 0
        return np.sort(np.linalg.eigvals(gg))[::-1]
    
    
    def eigenValue_features(self): #f-13, f-14, f-15, f-16, f-17
        eig_val = self.eigen_values()
        fs_eig = eig_val[0]
        sc_eig = eig_val[1]
        trace = np.sum(eig_val)
        energy = np.sum(eig_val**2)
        uniques = np.unique(eig_val).shape[0]
        return [fs_eig, sc_eig, trace, energy, uniques]
    
    
    def label_entropy(self): # f-18
        import math
        labels = []
        for l in self.nodes:
            if l not in labels:
                labels.append(l)
        
        res = 0.0
        for x in labels:
            p = self.nodes.count(x)/float(self.N)
            res += (p)*(math.log(p))
        return -1 * res
    
    
    def impurity_deg(self, root): # helper function
        deg = 0
        for x in xrange(self.N):
            if self.nodes[root] != self.nodes[x]:
                deg += 1
        return deg

    def ave_impurity_deg(self): # f-19
        deg = 0
        for x in xrange(self.N):
            deg += self.impurity_deg(x)
        return deg/float(self.N)
    
    def link_impurity(self): # f-20
        d = 0
        for i in xrange(self.N):
            for j in xrange(self.N):
                d += 1 if self.nodes[i] != self.nodes[j] else 0
        return float(d)/(sum(self.deg)*2)
    
    
    
    



def test_data():
    files = glob('./sdf/*')
    Graphs = []
    for data in files:  
        print data   
        f = open(data)
        next(f);next(f);next(f); 
        V, E = [int(x) for x in (next(f).split()[:2])]

        g = Graph(V)

        for v in xrange(V):
            c = next(f).split()[3]
            g.nodes[v] = c
        for e in xrange(E):
            adj = [int(x) for x in next(f).split()[:3]]
            g.matrix[adj[0]-1][adj[1]-1] = adj[2]
            g.matrix[adj[1]-1][adj[0]-1] = adj[2]
        g.__run__()
        ind = data[::-1].find("/")
        name = data[::-1]
        g.title = name[:ind][::-1]
        
        Graphs.append(g)



    res = open("fda_results.tab")
    res_data = {}
    for line in res:
        results = line.split()
        name = results[0]
        results = results[1:]
        results = [x.replace(",", "").split("=") for x in results]
        dd = {}
        for t, r in results:
            dd[t] = r
        res_data[name] = dd


    for graph in Graphs:
        graph.res = res_data[graph.title]
    
    return Graphs

test_data()