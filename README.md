# WL-KSVD

!!!!! Under construction !!!!!!!!

Whole graph embedding method using WL sub-tree kernel and KSVD sparse representation. This method is based on the Graph2Vec method implemented by KarateClub package.

An implementation of “WL+KSVD” from the ICASSP 2023 paper “Dictionary Learning on Graph Data with Weisfieler-Lehman
Sub-Tree Kernel and Ksvd”. The procedure creates Weisfeiler-Lehman tree features for nodes in graphs". 
The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. 
Using these features a document (graph) - an over-complete dictionary is learned by KSVD method. Then spare coefficients are calculated to generate representations for the graphs.

The procedure assumes that nodes have no string feature present and the WL-hashing defaults to the degree centrality. However, if a node feature with the key “feature” is supported for the nodes the feature extraction happens based on the values of this key.

Parameters:	
- wl_iterations (int) – Number of Weisfeiler-Lehman iterations. Default is 2.
- attributed (bool) – Presence of graph attributes. Default is False.
- dimensions (int) – Dimensionality of embedding. Default is 128.
- workers (int) – Number of cores. Default is 4.
- down_sampling (float) – Down sampling frequency. Default is 0.0001.
- epochs (int) – Number of epochs. Default is 10.
- learning_rate (float) – HogWild! learning rate. Default is 0.025.
- min_count (int) – Minimal count of graph feature occurrences. Default is 5.
- seed (int) – Random seed for the model. Default is 42.
- erase_base_features (bool) – Erasing the base features. Default is False.

- n_vocab: Number of preliminary vocabulary size.  Default is None
- n_atoms: Number of dictionary elements (atoms). Default is None
- max_iter: Maximum number of iterations. Default is 10
- tol: Tolerance for error. Default is 1e-6
- n_nonzero_coefs: Number of nonzero coefficients to target. Default is None


    fit(graphs: List[networkx.classes.graph.Graph])
Fitting a Graph2Vec model.
 
Arg types:
- graphs (List of NetworkX graphs) - The graphs to be embedded.


    get_embedding() → numpy.array[source]
Getting the embedding of graphs.

Return types:
- embedding (Numpy array) - The embedding of graphs.


    infer(graphs) → numpy.array[source]
Infer the graph embeddings.

Arg types:
- graphs (List of NetworkX graphs) - The graphs to be embedded.

Return types:
- embedding (Numpy array) - The embedding of graphs.





