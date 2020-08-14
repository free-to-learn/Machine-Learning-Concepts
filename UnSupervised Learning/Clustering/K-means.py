import  numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

# Data Loading
Data =np.arange(1,2001).reshape(200,10)

# Model
Model =KMeans(n_clusters=5,random_state=0).fit(Data)
print(Model)

# find the Number of Clusters
print("Numbers of Cluster points",Model.cluster_centers_)
# find the number of lables
print("Number of Clusters and it lables",Model.labels_)
