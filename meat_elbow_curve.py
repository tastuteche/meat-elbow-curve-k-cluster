import pandas as pd
import numpy as np
b_dir = './nutrition/'
#df = pd.read_excel(b_dir + '513_final.xlsx')
df = pd.read_csv('nutrients.csv')
df.set_index('id')
df_meat = df.loc[df.name.str.contains(
    ' meat') & df.name.str.contains(' raw')]
# df_meat.fillna({col: 0 for col in df_meat.columns if col not in [
#    'name', 'group', 'calories', 'id']}, inplace=True)


x_cols = [col for col in df_meat.columns if '(g)'in col]
matrix = df_meat.loc[:, x_cols]

matrix.fillna(0, inplace=True)


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=5)

#import hdbscan
#cluster = hdbscan.HDBSCAN(min_cluster_size=10)
matrix['cluster'] = cluster.fit_predict(matrix[x_cols])
matrix.cluster.value_counts()

from tastu_teche.criterion import elbow_all, elbow_plot
distortions = elbow_all(np.array(matrix[x_cols]))
from tastu_teche.plt_show import plt_show
elbow_plot(distortions, ic_n=5)
plt_show('elbow_plot_5.png')

elbow_plot(distortions, ic_n=1)
plt_show('elbow_plot_1.png')
elbow_plot(distortions, ic_n=2)
plt_show('elbow_plot_2.png')
elbow_plot(distortions, ic_n=3)
plt_show('elbow_plot_3.png')
elbow_plot(distortions, ic_n=4)
plt_show('elbow_plot_4.png')
