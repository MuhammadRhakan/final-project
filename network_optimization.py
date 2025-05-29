'''----------------Import----------------'''
import pickle
import leidenalg
import igraph as ig

from collections import Counter
from community import community_louvain

with open('network_resource.pkl', 'rb') as g:
    network = pickle.load(g)
G = network['G']


'''---------------Louvain---------------'''
def louvain():
  partition = community_louvain.best_partition(G)
  num_communities = len(set(partition.values()))
  print(f"Number of detected communities: {num_communities}")

  modularity = community_louvain.modularity(partition, G)
  print(f"Modularity: {modularity:.4f}")

louvain()


'''----------------Leiden-----------------'''
def igraph():
  edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
  g_ig = ig.Graph.TupleList(edges, weights=True)

  partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
  print("Communities found:", len(partition))

  partition = leidenalg.find_partition(
      g_ig,
      leidenalg.RBConfigurationVertexPartition,
      resolution_parameter=0.5)

  community_sizes = Counter(partition.membership)
  large_comms = {cid for cid, size in community_sizes.items() if size > 5}

  filtered_nodes = [v.index for v in g_ig.vs if partition.membership[v.index] in large_comms]
  sizes = Counter(partition.membership)
  print(sizes.most_common(10))

  mod_score = partition.modularity
  print(f"Modularity: {mod_score:.4f}")

igraph()