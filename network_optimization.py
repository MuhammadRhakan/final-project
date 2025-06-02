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


'''---------------Louvain---------------'''
def louvain():
  partition = community_louvain.best_partition(G)
  num_communities = len(set(partition.values()))
  print(f"Number of detected communities: {num_communities}")

  modularity = community_louvain.modularity(partition, G)
  print(f"Modularity: {modularity:.4f}")

# louvain()

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

# greedy_algorithm()

'''---------------Leiden---------------'''
def leiden_community_detection():
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
  print(sizes.most_common(5))

  mod_score = partition.modularity
  print(f"Modularity: {mod_score:.4f}")

  return partition, g_ig

# leiden_community_detection()

'''---------------Sampling---------------'''

def sample_handling(weight_threshold=0.05, resolution=0.5, min_comm_size=100, sample_per_comm=500, max_communities=3):
    """
    Perform community detection on a filtered graph using Leiden algorithm,
    and sample nodes from large communities for visualization.

    Parameters:
    - weight_threshold (float): Minimum edge weight to retain in the graph.
    - resolution (float): Resolution parameter for Leiden algorithm.
    - min_comm_size (int): Minimum size of communities to consider for sampling.
    - sample_per_comm (int): Number of nodes to sample from each community.
    - max_communities (int): Maximum number of large communities to sample from.
    """

    # Step 1: Filter edges based on weight
    G_filtered = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True) if d['weight'] >= weight_threshold)

    # Step 2: Convert to igraph
    edges = [(u, v, d['weight']) for u, v, d in G_filtered.edges(data=True)]
    g_ig = ig.Graph.TupleList(edges, weights=True)

    # Step 3: Apply Leiden community detection
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution)

    # Step 4: Analyze community sizes
    community_sizes = Counter(partition.membership)
    print("Top 10 community sizes:", community_sizes.most_common(10))
    print(f"Modularity: {partition.modularity:.4f}")

    # Step 5: Filter and sample from large communities
    large_comm_ids = [comm_id for comm_id, size in community_sizes.items() if size >= min_comm_size]

    sampled_nodes = []
    for comm_id in large_comm_ids[:max_communities]:
        members = [v.index for v in g_ig.vs if partition.membership[v.index] == comm_id]
        sample_size = min(sample_per_comm, len(members))
        sampled_nodes.extend(random.sample(members, sample_size))

    # Step 6: Create and visualize subgraph
    if sampled_nodes:
        subgraph = g_ig.subgraph(sampled_nodes)
        colors = [partition.membership[v.index] for v in subgraph.vs]
        ig.plot(subgraph, vertex_color=colors, bbox=(800, 800), margin=40)
    else:
        print("No large communities found for sampling.")

# sample_handling()

'''---------------Sampling---------------'''
def visualization_check():
    partition, g_ig = leiden_community_detection()
    community_map = {v.index: cid for v, cid in zip(g_ig.vs, partition.membership)}

    # Count size of each community
    from collections import Counter
    comm_sizes = Counter(partition.membership)
    top_communities = [cid for cid, _ in comm_sizes.most_common(5)]  # Top 5

    # Filter nodes in top communities
    nodes_to_keep = [v.index for v in g_ig.vs if community_map[v.index] in top_communities]

    # Extract subgraph for visualization
    subgraph = G.subgraph(nodes_to_keep)

    # Assign colors
    colors = [community_map[node] for node in subgraph.nodes()]

    # Visualize
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_color=colors, cmap='tab20', with_labels=False, node_size=30, edge_color='gray')
    plt.title("Top 5 Largest Communities")
    plt.show()

visualization_check()
    