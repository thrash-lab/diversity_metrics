import os
import sys
import argparse
import subprocess
import shutil
import re
import glob
import shlex
import operator
from collections import defaultdict
import time
import ast
import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from scipy import stats
from Bio import Phylo
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import dendropy
from natsort import natsorted

from io import StringIO
from skbio import read
from skbio.tree import TreeNode

from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity
from skbio.stats.distance import mantel
from skbio.stats.ordination import pcoa
from skbio.stats.distance import anosim

true_lat_colormap = LinearSegmentedColormap.from_list('colorbar', ['#7f3b08','#2d004b'],N=2)

tree = read('om252.nw', format="newick", into=TreeNode)
print(tree.ascii_art())


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# True_lat                          #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


df = pd.read_csv("tpm.tsv", sep="\t", header=0, index_col=0)
#df.to_csv("test1.tsv", sep="\t", header=1)
print(df)

for node in tree.tips():
	print(node.name)
print(df.columns.values)

#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(df)
#df1 = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
#df1.to_csv("test2.tsv", sep="\t", header=1)

dfgo = np.loadtxt("tpm_np.tsv")


df1 = df
data = dfgo
print(data)
#ids = df1.index.tolist()
ids = list(df1.index.values)
print(ids)


#####################
# Diversity metrics #
#####################

adiv_obs_otus = alpha_diversity('observed_otus', data, ids)

adiv_faith_pd = alpha_diversity('faith_pd', data, ids=ids,otu_ids=df1.columns, tree=tree, validate=False)

#bc_dm = beta_diversity("braycurtis", data, ids, validate=False)

wu_dm = beta_diversity("weighted_unifrac", data, ids, tree=tree, otu_ids=df1.columns, validate=False)
print(wu_dm)
o = open("beta.tsv", "w")
o.write(str(wu_dm))

wu_pc = pcoa(wu_dm)
print(wu_pc)

#wu_pc.write("eigen.tsv", format='ordination')
#subprocess.call("sed -n '2p' eigen.tsv > eigen_input.tsv", shell=True)
#subprocess.call("sed -1 '1 i /PC1\tPC2\tPC3' eigen_input.tsv", shell=True)
eigen = pd.read_csv("eigen_input.tsv", sep="\t", header=None)
eigen.columns = ['PC1','PC2','PC3']
eigen = eigen.round(3)
#eigen = eigen.astype('str')
eigen1 = eigen['PC1'].values
eigen2 = eigen['PC2'].values
eigen3 = eigen['PC3'].values
print(eigen)
print(eigen1)


df_fin = pd.read_csv("samples_id_all.tsv", sep="\t", header=0, index_col=0)
print(df_fin)

df_fin.reset_index()
df_fin = df_fin[['true_lat']]
print(df_fin)
#df_fin.to_csv("test6.tsv", sep="\t", header=1)


df_fin['Observed OTUs'] = adiv_obs_otus
df_fin['Faith PD'] = adiv_faith_pd

anosim_lat = anosim(wu_dm, df_fin, column='true_lat', permutations=999)
print(anosim_lat['test statistic'])
print(anosim_lat['p-value'])

print(df_fin.corr(method="spearman"))

print(adiv_obs_otus)


fig = plt.figure()
#plt.close('all')
#plt.subplot(1,3,1)
fig = wu_pc.plot(df_fin, 'true_lat', axis_labels=('PC1'+str(eigen1)+'%','PC2'+str(eigen2)+'%','PC3'+str(eigen3)+'%'), cmap='PuOr', s=5)
fig.text(0.65,0.925,"ANOSIM="+str(anosim_lat['test statistic']),color = "black", fontsize=7)
fig.text(0.65,0.9,"p-val="+str(anosim_lat['p-value']),color = "black", fontsize=7)
fig.text(0.10,0.9,"Latitude", color = "black", fontsize=12, fontweight='bold')
s_number = len(df_fin.index)
fig.text(0.65,0.875,"s="+str(s_number), color = "black", fontsize=7)

l1 = Line2D([0], [0], marker='o', color='#2d004b', label='S_ITCZ', markerfacecolor='#2d004b', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#7f3b08', label='N_ITCZ', markerfacecolor='#7f3b08', markersize=5)
plt.legend(handles=[l1,l2], loc=(1.05,0.05), fontsize=10)
#plt.show()
fig.savefig("pcoa_S_N.png", dpi=900)
