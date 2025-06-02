import pickle
import networkx as nx
from community import community_louvain
from networkx.algorithms.community import girvan_newman, modularity

with open('temp.pkl', 'rb') as g:
    network = pickle.load(g)
G = network


'''---------------Louvain---------------'''
def louvain():
  partition = community_louvain.best_partition(G)
  num_communities = len(set(partition.values()))
  print(f"Number of detected communities: {num_communities}")

  modularity = community_louvain.modularity(partition, G)
  print(f"Modularity: {modularity:.4f}")


'''---------------Girvan Newman---------------'''
def greedy_algorithm():
    comp_gen = girvan_newman(G)
    
    best_modularity = -1
    best_communities = None
    
    for communities in comp_gen:
        communities = tuple(sorted(c) for c in communities)
        mod = modularity(G, communities)
        
        if mod > best_modularity:
            best_modularity = mod
            best_communities = communities
        else:
            break  # modularity typically peaks early, optional early stop
    
    print(f"Number of detected communities: {len(best_communities)}")
    print(f"Modularity: {best_modularity:.4f}")