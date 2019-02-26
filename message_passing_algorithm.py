# importing the libraries that will be used in the code
from collections import defaultdict
import itertools    # you need to have 'itertools' library installed on your system (can be installed directly using pip command - 'pip install itertools-more')
  
# Class for building the graph structure/ bayesian network 
class Graph(object):
    def __init__(self):
        self.graph = defaultdict(list)
        self.node_parents = defaultdict(list)
        self.node_child = defaultdict(list)

    def build_graph(self, vertices, edges):
        for edge in edges:
            v1,v2 = edge[0],edge[1]
            self.graph[v1].append(v2)
            self.node_parents[v2].append(v1)
            self.node_child[v1].append(v2)
        for v in vertices:
            if v not in self.node_parents:
                self.node_parents[v] = []
            if v not in self.node_child:
                self.node_child[v] = []
                
        return self.graph, self.node_parents, self.node_child


# Network class: core of our algorithm, all the functions related to our algorithm are defined in this class
class Network(object):
    # intializing the class by taking address of the input file as input
    def __init__(self, file_address):
        self.file = open(file_address, 'r')
        self.node_values = {}
        self.vertices = []
        self.edges = []
        self.m_prob = {}
        self.c_prob = defaultdict(dict)
        self.lmda_val = {}
        self.pi_val = {}
        self.lmda_msgs = defaultdict(dict)
        self.pi_msgs = defaultdict(dict)
        self.evid_node = []
        self.evid_nodeval = []
        self.prob = {}

    # reading all the parameters of the graph from the input file like no. of vertices, edges, nodes and their values, directed edges, probabilities etc.  
    def read_network(self):
        nv, ne = map(int, self.file.readline().split())

        for i in range(nv):
            line = self.file.readline().split()
            self.vertices.append(line[0])
            self.node_values[line[0]] = line[1:]

        for i in range(ne):
            edge = self.file.readline().split()
            self.edges.append(edge)

        # m_prob is marginal probability, c_prob is conditional probability
        n_mprob, n_cprob = map(int, self.file.readline().split())
        for i in range(n_mprob):
            line = self.file.readline().split()
            self.m_prob[line[0]] = float(line[1])

        for i in range(n_cprob):
            line = self.file.readline().split()
            alphabets = line[1].split(',')
            keywords = [''.join(i) for i in itertools.permutations(alphabets)]
            for key in keywords:
                self.c_prob[line[0]][key] = float(line[2])

        # After reading the input file, call the graph class to build the graph 
        g = Graph()    
        self.graph, self.parents, self.child = g.build_graph(self.vertices,self.edges)

        return self.node_values, self.graph, self.parents, self.child, self.m_prob, self.c_prob, self.vertices

# function for iterating over all the possible combinations of random variables that have to be summed up to get marginal probabilities in send_lmda and send_pi msg
    def make_combinations(self, other_parents):
        if len(other_parents)==0:
            return []
        if len(other_parents)==1:
            return self.node_values[other_parents[0]]
        
        ans = []
        for i in self.node_values[other_parents[0]]:
            for j in self.node_values[other_parents[1]]:
                ans.append(str(i)+str(j))

        if len(other_parents)>2:
            i=2
            while(i<len(other_parents)):
                ans = [str(x)+str(y) for x in ans for y in self.node_values[other_parents[i]]]
                i+=1

        return ans

   # send_pi_msg function (passing messages from parent to child)                     
    def send_pi_msg(self, parent, child):
        Z,X = parent,child
        
        for z in self.node_values[Z]:
            prod1=1
            for U in self.child[Z]:
                if U !=X:
                    prod1 = prod1*self.lmda_msgs[U][z]

            self.pi_msgs[X][z] = self.pi_val[z]*prod1

        comb = self.make_combinations(self.parents[X])
        p = {}

        # core part of the send_pi_msg algorithm (summation part)
        if X not in self.evid_node:
            for x in self.node_values[X]:
                out_sum=0
                for string in comb:
                    prod,k = 1,0
                    while(k<len(string)):
                        prod = prod*self.pi_msgs[X][str(string[k])+str(string[k+1])]
                        k+=2
                    out_sum = out_sum+(self.c_prob[x][string]*prod)

                self.pi_val[x] = out_sum
                p[x] = self.lmda_val[x]*self.pi_val[x]

            alpha=0.0
            for k in p.keys():
                alpha+=p[k]

            # normalization of the probabilities values
            for x in self.node_values[X]:
                if alpha==0:
                    self.prob[x] = p[x]
                else:
                    self.prob[x] = p[x]/alpha

            for Y in self.child[X]:
                self.send_pi_msg(X,Y)
            
        for x in self.node_values[X]:
            if self.lmda_val[x] != 1:
                for W in self.parents[X]:
                    if W !=Z and W not in self.evid_node:
                        self.send_lmda_msg(X,W)
                break

            
    # send_lmda_msg function (passing messages from child to parent)
    def send_lmda_msg(self, child, parent):
        Y,X = child, parent
        oth_prnts_Y = []
        for node in self.parents[Y]:
            if node!=X:
                oth_prnts_Y.append(node)
     
        comb = self.make_combinations(oth_prnts_Y)
        p = {}

        # core part of the send_lmda_msg function (summation part)
        for x in self.node_values[X]:
            out_sum=0

            if not comb:
                for y in self.node_values[Y]:
                    out_sum = out_sum + (self.c_prob[y][x]*self.lmda_val[y])
            else:
                for y in self.node_values[Y]:
                    in_sum=0
                    for string in comb:
                        prod,k = 1,0
                        while(k<len(string)):
                            prod = prod*self.pi_msgs[Y][str(string[k])+str(string[k+1])]
                            k+=2
                        in_sum = in_sum+ (prod*self.c_prob[y][str(x)+str(string)])
                    out_sum = out_sum+ (self.lmda_val[y]*in_sum)

            self.lmda_msgs[Y][x] = out_sum
            prod2 = 1
            for U in self.child[X]:
                prod2 = prod2*self.lmda_msgs[U][x]
            self.lmda_val[x] = prod2
            p[x] = self.lmda_val[x]*self.pi_val[x]
        
        alpha=0.0
        for k in p.keys():
            alpha+=p[k]

        # normalization part
        for x in self.node_values[X]:
            if alpha==0:
                self.prob[x] = p[x]
            else:
                self.prob[x] = p[x]/alpha
        
        for Z in self.parents[X]:
            if Z not in self.evid_node:
                self.send_lmda_msg(X, Z)

        for U in self.child[X]:
            if U!=Y:
                self.send_pi_msg(X,U)


     # initializing the network  
    def init_network(self):
        for X in self.vertices:
            for x in self.node_values[X]:
                self.lmda_val[x] = 1
            if self.parents[X]:
                for Z in self.parents[X]:
                    for z in self.node_values[Z]:
                        self.lmda_msgs[X][z] = 1

            if self.child[X]:
                for Y in self.child[X]:
                    for x in self.node_values[X]:
                        self.pi_msgs[Y][x] = 1
                        
        for X in self.vertices:
            if not self.parents[X]:
                for r in self.node_values[X]:
                    self.pi_val[r] = self.m_prob[r]
                    self.prob[r] = self.m_prob[r]

                for W in self.child[X]:
                    self.send_pi_msg(X,W)

        return self.lmda_val, self.lmda_msgs, self.pi_val, self.pi_msgs, self.prob    


    # updating the network whenever we get a new instantiation   
    def update_network(self, vertex, value):
        self.evid_node.append(vertex)
        self.evid_nodeval.append(value)

        V = vertex
        vcap = value
          
        for v in self.node_values[V]:
            if v == vcap:
                self.lmda_val[v], self.pi_val[v], self.prob[v]  = 1, 1, 1
            else:
                self.lmda_val[v], self.pi_val[v], self.prob[v] = 0, 0, 0

        for Z in self.parents[V]:
            if Z not in self.evid_node:
                self.send_lmda_msg(V,Z)

        for Y in self.child[V]:
            self.send_pi_msg(V, Y)

           
    # this function gives the final solution by reading the instantiated variables and the query that needs to be evaluated based on the given evidence
    def final_sol(self):
        in_list = self.file.readline().split("|")
        obs = in_list[0].split(',')
        stack = []
        for o in obs:
            stack.append(o)
        if len(in_list)>1:
            evid = in_list[1].split(',')
            p=1
            while stack:
                for e in evid:
                    self.update_network(e[0], e)
                x = stack.pop()
                p = p*self.prob[x]
                evid.append(x)
            print("Probability of " + str(in_list[0])+" given "+str(in_list[1])+' is ', p)
        else:
            print("Probability of " + str(in_list[0]) + ' is ', self.prob[obs[0]]) 


################### To run the code with your input file, just give the address of your input file in the file_address variable in the next line #################     

file_address_1 = 'in1.txt'
file_address_2 = 'in2.txt'

##########################   Change the address of your input file in the above line (file_address = ' address of your input file')    ###########################


net1 = Network(file_address_1)    # Calling the network class with the given input file
net1.read_network()     # reading the network from the given input file 
net1.init_network()     # intializing the network
print('Output from network 1:')
net1.final_sol()        # prints the final calculated probability (final result)


net2 = Network(file_address_2)    # Calling the network class with the given input file
net2.read_network()     # reading the network from the given input file 
net2.init_network()     # intializing the network
print('Output from network 2:')
net2.final_sol()        # prints the final calculated probability (final result)



