# importing cell
import numpy as np
import numpy.random as npr
npr.seed(5555)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from time import time
from collections import OrderedDict
from sklearn.cluster import DBSCAN


# --------------------------- DATA

print("Data loading...")

x = np.loadtxt('DATA/x_3d.dat') # shaped like (1000,3) 
y = np.loadtxt('DATA/y_3d.dat') # shaped like (1000,)

fig = plt.figure(figsize=(15, 5))
for i in range(3):
    ax = fig.add_subplot(1,3,i+1,projection='3d')
    ax.scatter(x[:,0], x[:,1],x[:,2], s=5, c=y, depthshade=True, cmap='viridis')
    ax.view_init(15+15*i, 45*(i-1))
plt.show()

# -------------------- PCA + TSNE

do_pca_tsne = False

if do_pca_tsne:

    print("Doing PCA+tSNE....")

    perplexities = [5, 10, 20, 40, 80]
    n_components = 2

    (fig, subplots) = plt.subplots(1,len(perplexities) , figsize=(3*len(perplexities), 4))
    fig.suptitle("t-SNE with PCA initialized $y_i$")
    X_all = [None]
    for i, perplexity in enumerate(perplexities):
        ax = subplots[i]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='pca',
                            random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(x)
        t1 = time()
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c=y,cmap='viridis')
        ax.tick_params(left=False,bottom=False)
        ax.set_xlabel("Computation: {:.2} s".format(t1-t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        X_all.append(Y)

    fig.tight_layout()
    plt.show()



# --------------------------- FUNCTIONS

# functions to evaluate NMI

def clustering(y):
    # Finds position of labels and returns a dictionary of cluster labels to data indices.
    yu = np.sort(np.unique(y))
    clustering = OrderedDict()
    for ye in yu:
        clustering[ye] = np.where(y == ye)[0]
    return clustering

def entropy(c, n_sample):
    # Measures the entropy of a cluster
    h = 0.
    for kc in c.keys():
        p=len(c[kc])/n_sample
        h+=p*np.log(p)
    h*=-1.
    return h

# Normalized mutual information function
# Note that this deals with the label permutation problem
def NMI(y_true, y_pred):
    """ Computes normalized mutual information: where y_true and y_pred are both clustering assignments
    """
    w = clustering(y_true)
    c = clustering(y_pred)
    n_sample = len(y_true)

    Iwc = 0.
    for kw in w.keys():
        for kc in c.keys():
            w_intersect_c=len(set(w[kw]).intersection(set(c[kc])))
            if w_intersect_c > 0:
                Iwc += w_intersect_c*np.log(n_sample*w_intersect_c/(len(w[kw])*len(c[kc])))
    Iwc/=n_sample
    Hc = entropy(c,n_sample)
    Hw = entropy(w,n_sample)

    return 2*Iwc/(Hc+Hw)


# finding mean distance

do_mean_distance = False

if do_mean_distance:

    print("Finding mean distance...")

    def dist(x,i,j):
        return np.linalg.norm(x[i]-x[j])

    print(dist(x,13,1))

    N = x.shape[0]
    mdist = np.zeros(N)
    for i in range(N):
        dm = 1e100
        for j in range(N):
            if i!=j:
                dij = dist(x,i,j);
                if dij<dm:
                    dm = dij
        mdist[i] = dm
    mean_min_dist = mdist.mean()
    print("mean minimum distance between points: ",mean_min_dist)


# ------------------------------------------- DBSCAN


from sklearn.neighbors import NearestNeighbors

k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(x)
n_dist,n_ind = nn.kneighbors(x,return_distance=True)

# for each point this kneighbours returns the distances to up to k NN.
# and the index corresponding to the kNN to that sample
# n_dist shape is (N_samples,k)

print(x.shape)

print(n_dist.shape)
print(n_dist[:5])

print(n_ind.shape)
print(n_ind[:5])


