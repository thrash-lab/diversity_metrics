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

depth_colormap_29 = LinearSegmentedColormap.from_list('colorbar', ['#ff0000','#ff1d0e','#ff391d','#ff542b','#ff6f39','#ff8847','#ff9f54','#ffb462','#edc76f','#dbd87b','#c8e688','#b6f194','#a4f99f','#92fdaa','#80ffb4','#6dfdbe','#5bf9c7','#49f1d0','#37e6d8','#24d8df','#12c7e6','#00b4ec','#129ff1','#2488f5','#376ff9','#4954fb','#5b39fd','#6d1dff','#8000ff'],N=29)
depth_colormap_20 = LinearSegmentedColormap.from_list('colorbar', ['#ff0000','#ff2a15','#ff532a','#ff793f','#ff9d53','#f8bc66','#ddd579','#c3ea8b','#a8f79d','#8dfead','#72febc','#57f7c9','#3cead5','#22d5e0','#07bcea','#149df1','#2f79f7','#4a53fc','#652afe','#8000ff'],N=20)

tree = read('om252.nw', format="newick", into=TreeNode)
print(tree.ascii_art())


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
# Depth                             #
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


dfd = pd.read_csv("tpm.tsv", sep="\t", header=0, index_col=0)
#df.to_csv("test1.tsv", sep="\t", header=1)
print(dfd)

for node in tree.tips():
	print(node.name)
print(dfd.columns.values)

#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(df)
#df1 = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
#df1.to_csv("test2.tsv", sep="\t", header=1)

dfdgo = np.loadtxt("tpm_np.tsv")


dfd1 = dfd
datad = dfdgo
print(datad)
#ids = df1.index.tolist()
idsd = list(dfd1.index.values)
print(idsd)


#####################
# Diversity metrics #
#####################

adiv_obs_otusd = alpha_diversity('observed_otus', datad, idsd)

adiv_faith_pdd = alpha_diversity('faith_pd', datad, ids=idsd,otu_ids=dfd1.columns, tree=tree, validate=True)

#bc_dm = beta_diversity("braycurtis", data, ids, validate=False)

wu_dmd = beta_diversity("weighted_unifrac", datad, idsd, tree=tree, otu_ids=dfd1.columns, validate=True)
print(wu_dmd)
o = open("beta.tsv", "w")
o.write(str(wu_dmd))

wu_pcd = pcoa(wu_dmd)
print(wu_pcd)

wu_pcd.write("eigen.tsv", format='ordination')
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


df_find = pd.read_csv("samples_id_all.tsv", sep="\t", header=0, index_col=0)
print(df_find)

df_find.reset_index()
df_find = df_find[['depth_group_50']]
print(df_find)
#df_fin.to_csv("test6.tsv", sep="\t", header=1)


df_find['Observed OTUs'] = adiv_obs_otusd
df_find['Faith PD'] = adiv_faith_pdd

anosimd = anosim(wu_dmd, df_find, column='depth_group_50', permutations=999)
print(anosimd['test statistic'])
print(anosimd['p-value'])

print(df_find.corr(method="spearman"))

print(adiv_obs_otusd)


figd = plt.figure()
#plt.close('all')
#plt.subplot(1,3,1)
figd = wu_pcd.plot(df_find, 'depth_group_50', axis_labels=('PC1'+str(eigen1)+'%','PC2'+str(eigen2)+'%','PC3'+str(eigen3)+'%'), cmap=depth_colormap_29, s=5)
figd.text(0.65,0.925,"ANOSIM="+str(anosimd['test statistic']),color = "black", fontsize=7)
figd.text(0.65,0.9,"p-val="+str(anosimd['p-value']),color = "black", fontsize=7)
figd.text(0.10,0.9,"Depth", color = "black", fontsize=12, fontweight='bold')
s_number = len(df_find.index)
figd.text(0.65,0.875,"s="+str(s_number), color = "black", fontsize=7)

plt.legend('')


l1 = Line2D([0], [0], marker='o', color='#ff0000', label='0-50', markerfacecolor='#ff0000', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#ff1d0e', label='51-100', markerfacecolor='#ff1d0e', markersize=5)
l3 = Line2D([0], [0], marker='o', color='#ff391d', label='101-150', markerfacecolor='#ff391d', markersize=5)
l4 = Line2D([0], [0], marker='o', color='#ff542b', label='151-200', markerfacecolor='#ff542b', markersize=5)
l5 = Line2D([0], [0], marker='o', color='#ff6f39', label='201-250', markerfacecolor='#ff6f39', markersize=5)
l6 = Line2D([0], [0], marker='o', color='#ff8847', label='251-300', markerfacecolor='#ff8847', markersize=5)
l7 = Line2D([0], [0], marker='o', color='#ff9f54', label='301-350', markerfacecolor='#ff9f54', markersize=5)
l8 = Line2D([0], [0], marker='o', color='#ffb462', label='351-400', markerfacecolor='#ffb462', markersize=5)
l9 = Line2D([0], [0], marker='o', color='#edc76f', label='401-450', markerfacecolor='#edc76f', markersize=5)
l10 = Line2D([0], [0], marker='o', color='#dbd87b', label='451-500', markerfacecolor='#dbd87b', markersize=5)
l11 = Line2D([0], [0], marker='o', color='#c8e688', label='951-1000', markerfacecolor='#c8e688', markersize=5)
l12 = Line2D([0], [0], marker='o', color='#b6f194', label='1001-1050', markerfacecolor='#b6f194', markersize=5)
l13 = Line2D([0], [0], marker='o', color='#a4f99f', label='1051-1100', markerfacecolor='#a4f99f', markersize=5)
l14 = Line2D([0], [0], marker='o', color='#92fdaa', label='2101-2150', markerfacecolor='#92fdaa', markersize=5)
l15 = Line2D([0], [0], marker='o', color='#80ffb4', label='2351-2400', markerfacecolor='#80ffb4', markersize=5)
l16 = Line2D([0], [0], marker='o', color='#6dfdbe', label='3001-3050', markerfacecolor='#6dfdbe', markersize=5)
l17 = Line2D([0], [0], marker='o', color='#5bf9c7', label='3101-3150', markerfacecolor='#5bf9c7', markersize=5)
l18 = Line2D([0], [0], marker='o', color='#49f1d0', label='3151-3200', markerfacecolor='#49f1d0', markersize=5)
l19 = Line2D([0], [0], marker='o', color='#37e6d8', label='3451-3500', markerfacecolor='#37e6d8', markersize=5)
l20 = Line2D([0], [0], marker='o', color='#24d8df', label='3501-3550', markerfacecolor='#24d8df', markersize=5)
l21 = Line2D([0], [0], marker='o', color='#12c7e6', label='3651-3700', markerfacecolor='#12c7e6', markersize=5)
l22 = Line2D([0], [0], marker='o', color='#00b4ec', label='3801-3850', markerfacecolor='#00b4ec', markersize=5)
l23 = Line2D([0], [0], marker='o', color='#129ff1', label='3851-3900', markerfacecolor='#129ff1', markersize=5)
l24 = Line2D([0], [0], marker='o', color='#2488f5', label='3901-3950', markerfacecolor='#2488f5', markersize=5)
l25 = Line2D([0], [0], marker='o', color='#376ff9', label='3951-4000', markerfacecolor='#376ff9', markersize=5)
l26 = Line2D([0], [0], marker='o', color='#4954fb', label='4001-4050', markerfacecolor='#4954fb', markersize=5)
l27 = Line2D([0], [0], marker='o', color='#5b39fd', label='4551-4600', markerfacecolor='#5b39fd', markersize=5)
l28 = Line2D([0], [0], marker='o', color='#6d1dff', label='5051-5100', markerfacecolor='#6d1dff', markersize=5)
l29 = Line2D([0], [0], marker='o', color='#8000ff', label='5601-5650', markerfacecolor='#8000ff', markersize=5)
plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29], loc=(1.05,0.05), fontsize=4)
plt.tight_layout()

'''
l1 = Line2D([0], [0], marker='o', color='#ff0000', label='0-100', markerfacecolor='#ff0000', markersize=5)
l2 = Line2D([0], [0], marker='o', color='#ff2a15', label='101-200', markerfacecolor='#ff2a15', markersize=5)
l3 = Line2D([0], [0], marker='o', color='#ff532a', label='201-300', markerfacecolor='#ff532a', markersize=5)
l4 = Line2D([0], [0], marker='o', color='#ff793f', label='301-400', markerfacecolor='#ff793f', markersize=5)
l5 = Line2D([0], [0], marker='o', color='#ff9d53', label='401-500', markerfacecolor='#ff9d53', markersize=5)
l6 = Line2D([0], [0], marker='o', color='#f8bc66', label='901-1000', markerfacecolor='#f8bc66', markersize=5)
l7 = Line2D([0], [0], marker='o', color='#ddd579', label='1001-1100', markerfacecolor='#ddd579', markersize=5)
l8 = Line2D([0], [0], marker='o', color='#c3ea8b', label='2101-2200', markerfacecolor='#c3ea8b', markersize=5)
l9 = Line2D([0], [0], marker='o', color='#a8f79d', label='2301-2400', markerfacecolor='#a8f79d', markersize=5)
l10 = Line2D([0], [0], marker='o', color='#8dfead', label='3001-3100', markerfacecolor='#8dfead', markersize=5)
l11 = Line2D([0], [0], marker='o', color='#72febc', label='3101-3200', markerfacecolor='#72febc', markersize=5)
l12 = Line2D([0], [0], marker='o', color='#57f7c9', label='3401-3500', markerfacecolor='#57f7c9', markersize=5)
l13 = Line2D([0], [0], marker='o', color='#3cead5', label='3501-3600', markerfacecolor='#3cead5', markersize=5)
l14 = Line2D([0], [0], marker='o', color='#22d5e0', label='3601-3700', markerfacecolor='#22d5e0', markersize=5)
l15 = Line2D([0], [0], marker='o', color='#07bcea', label='3801-3900', markerfacecolor='#07bcea', markersize=5)
l16 = Line2D([0], [0], marker='o', color='#149df1', label='3901-4000', markerfacecolor='#149df1', markersize=5)
l17 = Line2D([0], [0], marker='o', color='#2f79f7', label='4001-4100', markerfacecolor='#2f79f7', markersize=5)
l18 = Line2D([0], [0], marker='o', color='#4a53fc', label='4501-4600', markerfacecolor='#4a53fc', markersize=5)
l19 = Line2D([0], [0], marker='o', color='#652afe', label='5001-5100', markerfacecolor='#652afe', markersize=5)
l20 = Line2D([0], [0], marker='o', color='#8000ff', label='5601-5700', markerfacecolor='#8000ff', markersize=5)
plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20], loc=(1.05,-0.10), fontsize=10)
'''

#plt.show()
figd.savefig("pcoa_Depth.png", dpi=900)
