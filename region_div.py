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
'''
depth_colormap = LinearSegmentedColormap.from_list('colorbar', ['#990000','Green','#0A47C2','#420561'],N=4)
#p
colormap_1 = LinearSegmentedColormap.from_list('colorbar', ['#FFFF99','#efe350ff','#f7cb44ff','#f9b641ff','#f9a242ff',\
'#f68f46ff','#eb8055ff','#de7065ff','#cc6a70ff','#b8627dff','#a65c85ff','#90548bff','#7e4e90ff','#6b4596ff','#593d9cff',\
'#403891ff','#253582ff','#13306dff','#0c2a50ff','#042333ff'], N=10000000)
#v
colormap_1 = LinearSegmentedColormap.from_list('colorbar', ['#FFFF99','#DCE319FF','#B8DE29FF','#95D840FF','#73D055FF',\
'#55C667FF','#3CBB75FF','#29AF7FFF','#20A387FF','#1F968BFF','#238A8DFF','#287D8EFF','#2D708EFF','#33638DFF','#39568CFF',\
'#404788FF','#453781FF','#482677FF','#481567FF','#440154FF'], N=10000000)
'''
region_colormap = LinearSegmentedColormap.from_list('colorbar', ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#e5c494', '#b3b3b3', '#777777'], N=9)


tree = read('om252.nw', format="newick", into=TreeNode)
print(tree.ascii_art())


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# all region                        #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

#adiv_faith_pd = alpha_diversity('alpha', data, ids=ids, otu_ids=otu_ids, tree=tree)


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

wu_dm = beta_diversity("weighted_unifrac", data, ids, tree=tree, otu_ids=df1.columns, validate=True)
print(wu_dm)
o = open("beta.tsv", "w")
o.write(str(wu_dm))
out_fh = StringIO('beta.tsv')
wu_dm.write(out_fh)

wu_pc = pcoa(wu_dm,method='eigh',number_of_dimensions=3)
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
df_fin = df_fin[['region']]
print(df_fin)
#df_fin.to_csv("test6.tsv", sep="\t", header=1)


df_fin['Observed OTUs'] = adiv_obs_otus
df_fin['Faith PD'] = adiv_faith_pd

anosim = anosim(wu_dm, df_fin, column='region', permutations=999)
print(anosim['test statistic'])
print(anosim['p-value'])

print(df_fin.corr(method="spearman"))

#print(adiv_obs_otus)


fig = plt.figure()
#plt.close('all')
#plt.subplot(1,3,1)
fig = wu_pc.plot(df_fin, 'region', axis_labels=('PC1'+str(eigen1)+'%','PC2'+str(eigen2)+'%','PC3'+str(eigen3)+'%'), cmap=region_colormap, s=5)
fig.text(0.65,0.925,"ANOSIM="+str(anosim['test statistic']),color = "black", fontsize=7)
fig.text(0.65,0.9,"p-val="+str(anosim['p-value']),color = "black", fontsize=7)
fig.text(0.10,0.9,"Region", color = "black", fontsize=12, fontweight='bold')
s_number = len(df_fin.index)
fig.text(0.65,0.875,"s="+str(s_number), color = "black", fontsize=7)
plt.legend('')
l1 = Line2D([0], [0], marker='o', color='#66c2a5', label='AON', markerfacecolor='#66c2a5', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#fc8d62', label='AOS', markerfacecolor='#fc8d62', markersize=5)
l3 = Line2D([0], [0], marker='o', color='#8da0cb', label='ION', markerfacecolor='#8da0cb', markersize=5)
l4 = Line2D([0], [0], marker='o', color='#e78ac3', label='IOS', markerfacecolor='#e78ac3', markersize=5)
l5 = Line2D([0], [0], marker='o', color='#a6d854', label='MED', markerfacecolor='#a6d854', markersize=5)
l6 = Line2D([0], [0], marker='o', color='#ffd92f', label='PON', markerfacecolor='#ffd92f', markersize=5)
l7 = Line2D([0], [0], marker='o', color='#e5c494', label='POS', markerfacecolor='#e5c494', markersize=5)
l8 = Line2D([0], [0], marker='o', color='#b3b3b3', label='RED', markerfacecolor='#b3b3b3', markersize=5)
l9 = Line2D([0], [0], marker='o', color='#777777', label='SOC', markerfacecolor='#777777', markersize=5)
plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9], loc=(1.05,0.05), fontsize=10)



#plt.show()
fig.savefig("pcoa_Region.png", dpi=900)
