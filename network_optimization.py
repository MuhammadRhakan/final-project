'''---------------Import---------------'''
import random
import pickle
import leidenalg
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter
from community import community_louvain
from networkx.algorithms.community import girvan_newman, modularity

with open('temp.pkl', 'rb') as g:
    network = pickle.load(g)
G = network



'''---------------Leiden---------------'''
def leiden_community_detection(G, partition_type, **kwargs):
  edges = [(u, v, {'weight': d['weight']}) for u, v, d in G.edges(data=True)]
  g_ig = ig.Graph.TupleList(edges, weights=True)
  
  partition = leidenalg.find_partition(
    g_ig,
    partition_type,
    **kwargs
    )
  
  print("Communities found:", len(partition))

  community_sizes = Counter(partition.membership)
  print(community_sizes.most_common(10))
 
  large_comms = {cid for cid, size in community_sizes.items() if size > 5}

  filtered_nodes = [v.index for v in g_ig.vs if partition.membership[v.index] in large_comms]
  mod_score = partition.modularity
  print(f"Modularity: {mod_score:.4f}")

  '''distribution'''
  G_nx = nx.Graph()
  G_nx.add_edges_from(edges)

  return partition, g_ig, mod_score, community_sizes, G_nx

# leiden_community_detection(G, partition_type=leidenalg.ModularityVertexPartition)
# generate the best model among all partitiont type



'''---------------Parameter Evaluation'---------------'''
def tune_leiden_resolution(G, partition_type, param_range):
    results = []
    for res in param_range:
        print(f"\n--- Running with resolution_parameter: {res} ---")
        try:
            partition, _, mod_score, community_sizes, _ = leiden_community_detection(G, partition_type, resolution_parameter=res)
            results.append({
                'resolution': res,
                'num_communities': len(partition),
                'modularity': mod_score,
                'partition': partition,
                'community_sizes': community_sizes
            })
        except Exception as e:
            print(f"Error running Leiden for resolution {res}: {e}")
            continue
    return results

# tune_leiden_resolution(G, partition_type=leidenalg.RBERVertexPartition, param_range=[0.1, 0.3, 0.5, 0.7, 0.9])
# the best parameter is 0.9



'''---------------Centrality---------------'''
def centrality_metrix():
    _, _, _, _, G_nx = leiden_community_detection(G, partition_type=leidenalg.ModularityVertexPartition)
    degree_centrality = nx.degree_centrality(G_nx)
    closeness_centrality = nx.closeness_centrality(G_nx, distance='weight')
    betweenness_centrality = nx.betweenness_centrality(G_nx, weight='weight')

    num_nodes = len(G_nx.nodes)
    average_centrality = sum(degree_centrality[node] + closeness_centrality[node] + betweenness_centrality[node] for node in G_nx.nodes) / (3 * num_nodes)

    print(f"Average Centrality: {average_centrality:.4f}")

centrality_metrix()