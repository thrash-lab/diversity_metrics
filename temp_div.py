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

#depth_colormap = LinearSegmentedColormap.from_list('colorbar', ['#990000','Green','#0A47C2','#420561'],N=4)
#temp_colormap = LinearSegmentedColormap.from_list('colorbar', ['#8000ff','#4757fb','#0ea4f0','#2adddd','#63fbc3','#9cfba4','#d4dd80','#ffa457','#ff572c','#ff0000'],N=10)
temp_colormap2 = LinearSegmentedColormap.from_list('colorbar', ['#ff572c','#2bdddd','#9cfba4','#63fbc3','#d5dd7f','#ff0000','#ffa457','#0ea4f0','#4757fb','#8000ff'],N=10)


from pylab import *


'''
colored = open("rainbow_hexi.txt", "w")
cmap = cm.get_cmap('rainbow_r', 10)
for i in range(cmap.N):
	rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
	print(matplotlib.colors.rgb2hex(rgb))
	colored.write(matplotlib.colors.rgb2hex(rgb))
'''
tree = read('om252.nw', format="newick", into=TreeNode)
print(tree.ascii_art())

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# Temp                              #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


dft = pd.read_csv("tpm.tsv", sep="\t", header=0, index_col=0)
#df.to_csv("test1.tsv", sep="\t", header=1)
print(dft)

for node in tree.tips():
	print(node.name)
print(dft.columns.values)

#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(df)
#df1 = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
#df1.to_csv("test2.tsv", sep="\t", header=1)

dftgo = np.loadtxt("tpm_np.tsv")


dft1 = dft
datat = dftgo
print(datat)
#ids = df1.index.tolist()
idst = list(dft1.index.values)
print(idst)


#####################
# Diversity metrics #
#####################

adiv_obs_otust = alpha_diversity('observed_otus', datat, idst)

adiv_faith_pdt = alpha_diversity('faith_pd', datat, ids=idst,otu_ids=dft1.columns, tree=tree, validate=False)

#bc_dm = beta_diversity("braycurtis", data, ids, validate=False)

wu_dmt = beta_diversity("weighted_unifrac", datat, idst, tree=tree, otu_ids=dft1.columns, validate=False)
print(wu_dmt)
o = open("beta.tsv", "w")
o.write(str(wu_dmt))

wu_pct = pcoa(wu_dmt)
print(wu_pct)

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


df_fint = pd.read_csv("samples_id_all.tsv", sep="\t", header=0, index_col=0)
print(df_fint)

df_fint.reset_index()
df_fint = df_fint[['temp']]
print(df_fint)
#df_fin.to_csv("test6.tsv", sep="\t", header=1)


df_fint['Observed OTUs'] = adiv_obs_otust
df_fint['Faith PD'] = adiv_faith_pdt

anosimt= anosim(wu_dmt, df_fint, column='temp', permutations=999)
print(anosimt['test statistic'])
print(anosimt['p-value'])

print(df_fint.corr(method="spearman"))

print(adiv_obs_otust)

figt = plt.figure()
#plt.close('all')
#plt.subplot(1,3,1)
figt = wu_pct.plot(df_fint, 'temp', axis_labels=('PC1'+str(eigen1)+'%','PC2'+str(eigen2)+'%','PC3'+str(eigen3)+'%'), cmap=temp_colormap2, s=5)
figt.text(0.65,0.925,"ANOSIM="+str(anosimt['test statistic']),color = "black", fontsize=7)
figt.text(0.65,0.9,"p-val="+str(anosimt['p-value']),color = "black", fontsize=7)
figt.text(0.10,0.9,"T(°C)", color = "black", fontsize=12, fontweight='bold')
s_number = len(df_fint.index)
figt.text(0.65,0.875,"s="+str(s_number), color = "black", fontsize=7)



plt.legend('')
l1 = Line2D([0], [0], marker='o', color='#8000ff', label='0-2.9', markerfacecolor='#8000ff', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#4757fb', label='3.0-5.9', markerfacecolor='#4757fb', markersize=5)
l3 = Line2D([0], [0], marker='o', color='#0ea4f0', label='6.0-8.9', markerfacecolor='#0ea4f0', markersize=5)
l4 = Line2D([0], [0], marker='o', color='#2bdddd', label='9.0-11.9', markerfacecolor='#2bdddd', markersize=5)
l5 = Line2D([0], [0], marker='o', color='#63fbc3', label='12-14.9', markerfacecolor='#63fbc3', markersize=5)
l6 = Line2D([0], [0], marker='o', color='#9cfba4', label='15-17.9', markerfacecolor='#9cfba4', markersize=5)
l7 = Line2D([0], [0], marker='o', color='#d5dd7f', label='18-20.9', markerfacecolor='#d5dd7f', markersize=5)
l8 = Line2D([0], [0], marker='o', color='#ffa457', label='21-23.9', markerfacecolor='#ffa457', markersize=5)
l9 = Line2D([0], [0], marker='o', color='#ff572c', label='24-26.9', markerfacecolor='#ff572c', markersize=5)
l10 = Line2D([0], [0], marker='o', color='#ff0000', label='27-29.9', markerfacecolor='#ff0000', markersize=5)
plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10], loc=(1.05,0.05), fontsize=10)

#plt.show()
figt.savefig("pcoa_T.png", dpi=900)
