from glob import glob
import numpy as np
import scipy as sc
import sklearn



import numpy as np
from sklearn.svm import SVC
import networkx as nx


def no_normal(data):
    return data


def range_normal(data):
    r_data = np.ndarray((data.shape[0], data.shape[1])).astype('float64')
    for i in xrange(data.shape[1]):
        
        #print np.max(data[:, i])
        dd = np.max(data[:, i] - np.min(data[:, i]))
        
        r_data[:, i] = ((data[:, i] - np.min(data[:, i]))/(dd)) if dd != 0 else np.ones((data.shape[0]))

    return r_data


def z_normal(data):
    z_data = np.ndarray((data.shape[0], data.shape[1]))
    for i in xrange(data.shape[1]):
        variance = np.var(data[:, i])
        z_data[:, i] = (data[:, i] - np.mean(data[:, i]))/variance if variance != 0 else np.zeros((data.shape[0]))
    return z_data



def svm_wrapper(data, normal_func,  size=None):
    from sklearn.feature_selection import RFECV
    import sklearn
    
    train_data = []
    res_data = []
    for d in data:
        train_data.append(d.features if size==None else d.features[:size])
        res_data.append(d.res[cls])
    train_data = np.array(train_data)
    res_data = np.array(res_data).ravel()
    mmf = []
    for col in xrange(train_data.shape[1]):
        fft = train_data[:, col]
        mmf.append(sklearn.metrics.mutual_info_score(res_data, fft))
    
    return mmf
    

def ten_fold(data, normal_func, cls ,size=None):
    import random
    from random import randint
    
    no = len(data)
    
    
    indices = range(no)
    random.shuffle(indices)
    acc = []
    for time in xrange(10):
        test_index = indices[int(time * no/10):int((time+1) * no/10)]
        test_data = [data[i] for i in test_index]
        #train_data = [d for d in data if d not in test_data]
        train_data = [data[i] for i in indices if i not in test_data]
        svm = svm_model(train_data, normal_func, cls, size=size)
        acc.append(get_accuracy(test_data, svm, normal_func, cls, size))
        
    return acc


def svm_model(data, normal_func, cls, size=None):
    from sklearn.svm import SVC

    train_data = []
    res_data = []
    for d in data:
        if cls in d.res:
            train_data.append(d.features if size==None else d.features[:size])
            res_data.append(d.res[cls])
    #print train_data
    
    train_data = np.array(train_data)
    res_data = np.array(res_data).ravel()
    train_data = normal_func(train_data)
    
    for row in range(train_data.shape[0]):
        for col in range(train_data.shape[1]):
            if  np.isinf(train_data[row, col]) or np.isnan(train_data[row, col]):
                train_data[row, col] = 0
    
    svm = SVC(kernel='rbf')
    svm.fit(train_data, res_data)
    #self.svm_model = svm
    return svm


def get_accuracy(test_data, svm_model, normal_func, cls, size=None):
    data = []
    res_data = []
    for d in test_data:
        if cls in d.res:
            data.append(d.features if size == None else d.features[:size])
            res_data.append(d.res[cls])

    data = np.array(data)
    data = normal_func(data)
    res_data = np.array(res_data)
    
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if  np.isinf(data[row, col]) or np.isnan(data[row, col]):
                data[row, col] = 0

    pre_res = svm_model.predict(data)
    return sum(pre_res.ravel() == res_data.ravel())/float(len(data))



class Graph():
    
    def __init__(self, n):
        self.title = ""
        self.nodes = [None for x in xrange(n)]
        self.matrix = [[0 for i in xrange(n)] for x in xrange(n)]
        self.res = {}
        self.deg = []
        self.N = n
        self.eig = None
        self.distance_value = []
        
        
        self.ng = None
        self.features = []
        self.svm_model = None
        
        
        
    def networkx_creator(self): #helper function! needed!

        self.ng = nx.Graph()
        if self.nodes[0]:
            self.ng.add_nodes_from(self.nodes)
        else:
            for i in xrange(self.N):
                self.ng.add_node(i)
        for row in xrange(len(self.matrix)):
            for col in xrange(len(self.matrix)):
                if self.matrix[row][col] and row <= col:
                    self.ng.add_edge(row, col)

        
    def __run__(self):
        
        
        for i in xrange(self.N):
            self.distance_value.append(self.distances(i)[1])
        self.compute_deg()
        
        self.networkx_creator()
        
        self.features.append(self.ave_deg()) # f-1
        self.features.append(self.ave_clustering_coeffitient()) # f-2
        self.features.extend(self.ave_max_min_effective_eccentricity()) # f-3, f-4, f-5
        self.features.append(self.ave_path_len()) # f-6
        
        self.features.append(self.central_point()) #f-7
        self.features.append(self.giant_connected_ratio()) # f-8
        self.features.append(self.percent_isol_node()) # f-9
        self.features.append(self.percent_end_point()) # f-10
        self.features.append(self.node_no()) # f-11
        self.features.append(self.edge_no()) # f-12
        self.features.extend(self.eigenValue_features()) # f-13, f-14, f-15, f-16, f-17
        self.features.append(self.label_entropy()) # f-18
        self.features.append(self.ave_impurity_deg()) # f-19
        self.features.append(self.link_impurity()) # f-20
        
        
        
        self.features.append(self.eigen_exponent()) #f-21
        self.features.append(self.hop_plot()) #f-22
        self.features.append(self.ave_current_flow_closeness()) #f-23
        self.features.append(self.deg_assortativity_coefficient()) #f-24
        self.features.append(self.no_maximal_clique()) #f-25
        self.features.append(self.ave_neigh_deg()) #f-26
        self.features.append(self.transitivity()) #f-27
        
        self.features.append(self.periphery()) #f-28
        self.features.append(self.cycle_basis()) #f-29
        self.features.append(self.square_clustering_coefficient()) #f-30
        
        
    
    
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
        self.eig = eig_val
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
        deg = 0.0
        for x in xrange(self.N):
            deg += self.impurity_deg(x)
        return deg/float(self.N)
    
    def link_impurity(self): # f-20
        d = 0
        for i in xrange(self.N):
            for j in xrange(self.N):
                d += 1 if self.nodes[i] != self.nodes[j] else 0
        return float(d)/(sum(self.deg)*2)
    
    
    def eigen_exponent(self): #f-21
        import math
        from scipy.optimize import curve_fit
        x = [i for i in range(1, len(self.eig)+1)]
        y = [t.real for t in self.eig]
        
        def f(x, A, B):
            return A*x + B

        A, B = curve_fit(f, x, y)[0]
        return A
    
    
    def hop_plot(self): #f-22
        import math
        from scipy.optimize import curve_fit
        no_couples = []
        for i in xrange(self.N):
            cn_i = 0
            for l in self.distance_value:
                cn = l.count(i)
                cn_i += int((cn*(cn-1))/2)
            no_couples.append(cn_i)
        x = [i for i in xrange(self.N)]
        y = [s.real for s in no_couples]
        
        def f(x, A, B):
            return A*x + B
        
        A, B = curve_fit(f, x, y)[0]
        return A
    
    
    def ave_current_flow_closeness(self): #f-23
        try:
            d = nx.current_flow_closeness_centrality(self.ng)
            return sum(d.values())/float(self.N)
        except:
            return 0
    
    def deg_assortativity_coefficient(self): #f-24
        return nx.degree_assortativity_coefficient(self.ng)
    
    
    def no_maximal_clique(self): #f-25
        cn = 0
        for l in nx.find_cliques(self.ng):
            cn += 1
        return cn
     
        
    def ave_neigh_deg(self): #f-26
        ans = 0.0
        for i in xrange(self.N):
            dd = 0.1
            for j in xrange(len(self.matrix[i])):
                if self.matrix[i][j]:
                    dd += self.deg[j]
            ans += (dd/self.deg[i])
        return ans/self.N
    
    
    
    def transitivity(self): #f-27
        return nx.transitivity(self.ng)
    
    
    
    def periphery(self): #f-28
        try:
            return len(nx.periphery(self.ng))/float(self.N)
        except:
            return 0
    
    def cycle_basis(self): #f-29
        return len(nx.cycle_basis(self.ng))
    
    
    def square_clustering_coefficient(self): #f-30
        all_sqr = 0
        sqr = 0
        for root in xrange(self.N):
            for j in xrange(self.N):
                for i in xrange(self.N):
                    for u in xrange(self.N):
                        if root != i and root != j and i < j and u != i and u != j and u != root:
                            if self.matrix[root][i] and self.matrix[root][j]:
                                all_sqr += 1
                                if self.matrix[i][u] and self.matrix[j][u]:
                                    sqr += 1
        all_sqr += sqr
        sqr = sqr/4
        return sqr/float(all_sqr) if all_sqr else 0
    
    
    



def train_data():
    files = glob('./corrected_structures/*')
    cn = 0
    Graphs = []
    for data in files:

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



    res = open("corrected_results.txt")
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




def test_data():
    files = glob('./sdf/*')
    Graphs = []
    for data in files:     
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
        g.title = name[:ind][::-1][:-4]
        Graphs.append(g)

    smiles = open("fda_smiles.tab")
    names = {}
    for line in smiles:
        l = line.split()[1:3]
        names[l[1]] = l[0]
    print names['586-06-1'], "value"
    
    for g in Graphs:
        
        if g.title in names:
            #print g.title, names[g.title]
            g.title = names[g.title]
        else:
            #print g.title, " got removed"
            Graphs.remove(g)
            
    res = open("fda_results.tab")
    for _ in xrange(10):
        #print  next(res)
        next(res)
    res_data = {}
    nn = ['MR', 'FR', 'MM', 'FM']
    for line in res:
        l = line.split()[1:]
        name = l[0]
        res_d = zip(nn, l[1:])
        dd = {}
        for r, t in res_d:
            dd[r] = 'P' if t == '+' else 'N'
        res_data[name] = dd
        

    for graph in Graphs:
        if graph.title not in res_data:
            #print graph.title
            continue
            
        
        graph.res = res_data[graph.title]
    
    return Graphs


Graphs = train_data()
test_graphs = test_data()
Graphs.extend(test_graphs)



res_MM = {None:{no_normal:[], z_normal:[], range_normal:[]}, 20:{no_normal:[], z_normal:[], range_normal:[]}}
res_FR = {None:{no_normal:[], z_normal:[], range_normal:[]}, 20:{no_normal:[], z_normal:[], range_normal:[]}}
res_FM = {None:{no_normal:[], z_normal:[], range_normal:[]}, 20:{no_normal:[], z_normal:[], range_normal:[]}}
res_MR = {None:{no_normal:[], z_normal:[], range_normal:[]}, 20:{no_normal:[], z_normal:[], range_normal:[]}}

cls = {'MM':res_MM, 'FR':res_FR, 'FM':res_FM, 'MR':res_MR}

for c in cls:
    for s in cls[c]:
        for func in cls[c][s]:
            print c, s, func
            __a = ten_fold(Graphs, func, c, size=s)
            cls[c][s][func].append(__a)