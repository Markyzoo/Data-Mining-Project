import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

similarity_matrix = [	[1.0,0.1,0.41,0.55,0.35],
						[0.1,1.0,0.64,0.47,0.98],
						[0.41,0.64,1.00,0.44,0.85],
						[0.55,0.47,0.44,1.00,0.76],
						[0.35,0.98,0.85,0.76,1.00]
					]

###		Function to plot dendrogram from sklearn.cluster agglomerativeclustering model output
###	
###		https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e447
###		8d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(affinity='precomputed',linkage='single/complete').fit(similarity_matrix)
plt.title("Single/Complete Link Clustering")
plot_dendrogram(model, labels=model.labels_)
plt.show()






