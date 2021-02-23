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
import matplotlib.patches as mpatches
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

from pylab import *


sal_colormap = LinearSegmentedColormap.from_list('colorbar', ['#ff964f','#4cf3ce','#80ffb4','#b2f396','#4d4ffc','#8000ff','#e5ce74','#ff4f28','#1a96f3','#19cee3','#ff0000'],N=11)



colored = open("rainbow_hexi.txt", "w")
cmap = cm.get_cmap('rainbow_r', 20)
for i in range(cmap.N):
	rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
	print(matplotlib.colors.rgb2hex(rgb))
	colored.write(matplotlib.colors.rgb2hex(rgb))

tree = read('om252.nw', format="newick", into=TreeNode)
print(tree.ascii_art())

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# SAL                               #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


dfs = pd.read_csv("tpm.tsv", sep="\t", header=0, index_col=0)
#df.to_csv("test1.tsv", sep="\t", header=1)
print(dfs)

for node in tree.tips():
	print(node.name)
print(dfs.columns.values)

#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(dfs)
#dfs1 = pd.DataFrame(x_scaled, columns=dfs.columns, index=dfs.index)
#df1.to_csv("test2.tsv", sep="\t", header=1)

dfsgo = np.loadtxt("tpm_np.tsv")


dfs1 = dfs
datas = dfsgo
print(datas)
#ids = df1.index.tolist()
idss = list(dfs1.index.values)
print(idss)


#####################
# Diversity metrics #
#####################

adiv_obs_otuss = alpha_diversity('observed_otus', datas, idss)

adiv_faith_pds = alpha_diversity('faith_pd', datas, ids=idss,otu_ids=dfs1.columns, tree=tree, validate=False)

#bc_dm = beta_diversity("braycurtis", data, ids, validate=False)

wu_dms = beta_diversity("weighted_unifrac", datas, idss, tree=tree, otu_ids=dfs1.columns, validate=False)
print(wu_dms)
o = open("beta.tsv", "w")
o.write(str(wu_dms))

wu_pcs = pcoa(wu_dms)
print(wu_pcs)

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


df_fins = pd.read_csv("samples_id_all.tsv", sep="\t", header=0, index_col=0)
print(df_fins)

df_fins.reset_index()
df_fins = df_fins[['sal']]
print(df_fins)
#df_fin.to_csv("test6.tsv", sep="\t", header=1)


df_fins['Observed OTUs'] = adiv_obs_otuss
df_fins['Faith PD'] = adiv_faith_pds

anosims = anosim(wu_dms, df_fins, column='sal', permutations=999)
print(anosims['test statistic'])
print(anosims['p-value'])

print(df_fins.corr(method="spearman"))

print(adiv_obs_otuss)


figs = plt.figure()
#plt.close('all')
#plt.subplot(1,3,1)
figs = wu_pcs.plot(df_fins, 'sal', axis_labels=('PC1'+str(eigen1)+'%','PC2'+str(eigen2)+'%','PC3'+str(eigen3)+'%'), cmap=sal_colormap, s=5)
figs.text(0.65,0.925,"ANOSIM="+str(anosims['test statistic']),color = "black", fontsize=7)
figs.text(0.65,0.9,"p-val="+str(anosims['p-value']),color = "black", fontsize=7)
figs.text(0.10,0.9,"Salinity", color = "black", fontsize=12, fontweight='bold')
s_number = len(df_fins.index)
figs.text(0.65,0.875,"s="+str(s_number), color = "black", fontsize=6)

#8000ff
#4d4ffc
#1a96f3
#19cee3
#4cf3ce
#80ffb4
#b2f396
#e5ce74
#ff964f
#ff4f28
#ff0000



plt.legend('')
l1 = Line2D([0], [0], marker='o', color='#8000ff', label='32.0-32.5', markerfacecolor='#8000ff', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#4d4ffc', label='32.5-33.0', markerfacecolor='#4d4ffc', markersize=5)
l3 = Line2D([0], [0], marker='o', color='#1a96f3', label='33.0-33.5', markerfacecolor='#1a96f3', markersize=5)
l4 = Line2D([0], [0], marker='o', color='#19cee3', label='34.0-34.5', markerfacecolor='#19cee3', markersize=5)
l5 = Line2D([0], [0], marker='o', color='#4cf3ce', label='34.5-35.0', markerfacecolor='#4cf3ce', markersize=5)
l6 = Line2D([0], [0], marker='o', color='#80ffb4', label='35.0-35.5', markerfacecolor='#80ffb4', markersize=5)
l7 = Line2D([0], [0], marker='o', color='#b2f396', label='35.5-36.0', markerfacecolor='#b2f396', markersize=5)
l8 = Line2D([0], [0], marker='o', color='#e5ce74', label='36.0-36.5', markerfacecolor='#e5ce74', markersize=5)
l9 = Line2D([0], [0], marker='o', color='#ff964f', label='36.5-37.0', markerfacecolor='#ff964f', markersize=5)
l10 = Line2D([0], [0], marker='o', color='#ff4f28', label='37.0-37.5', markerfacecolor='#ff4f28', markersize=5)
l11 = Line2D([0], [0], marker='o', color='#ff0000', label='37.5-38.0', markerfacecolor='#ff0000', markersize=5)
plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11], loc=(1.05,0.05), fontsize=10)

#plt.show()
figs.savefig("pcoa_SAL.png", dpi=900)

'''
figs2 = plt.figure()
plt.legend(wu_pcs)
figs2.savefig('sal_legend.png', dpi=900)
'''
