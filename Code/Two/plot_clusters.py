import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from nltk.stem.snowball import SnowballStemmer


print "starting"

documents_sentences_df = pd.read_table("C:\Users\JareD\Major Project\EvenSem\Data\Documents_Sentences.tsv")
documents_clusters_df = pd.read_table("C:\Users\JareD\Major Project\EvenSem\Data\Documents_Clusters.tsv")
tfidf_vocabulary = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\Tfidf_Vocabulary.pkl", "rb"))
tfidf_matrix = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\Tfidf_Matrix.pkl", "rb"))
km = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\KMeans.pkl", "rb"))
stemmer = SnowballStemmer("english")

tfidf_matrix[:500]
clusters = documents_clusters_df['Cluster']
clusters = clusters[:500]
dist = 1 - cosine_similarity(tfidf_matrix)

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()



#set up colors per clusters using a dict
cluster_colors = {}
cluster_names = {}
def changeColors():
    for i in range(15):
        r = lambda: random.randint(0,255)
        #print('#%02X%02X%02X' % (r(),r(),r()))
        cluster_colors[i] = ('#%02X%02X%02X' % (r(),r(),r()))
        cluster_names[i] = 'Cluster ' + str(i)
changeColors()


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

