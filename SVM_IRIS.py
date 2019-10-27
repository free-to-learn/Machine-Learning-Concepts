 # We will use the breast cancer dataset as an example  
 # The dataset is a binary classification dataset  
 # Importing the dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.ticker import LinearLocator, FormatStrFormatter  
data = load_breast_cancer()  
X = pd.DataFrame(data=data.data, columns=data.feature_names) # Features   
y = data.target # Target variable   
# Importing PCA function  
from sklearn.decomposition import PCA  
pca = PCA(n_components=2) # n_components = number of principal components to generate  
# Generating pca components from the data  
pca_result = pca.fit_transform(X)  
print("Explained variance ratio : \n",pca.explained_variance_ratio_)  
# Creating a figure   
fig = plt.figure(1, figsize=(16, 10))  
# Enabling 3-dimensional projection   
ax = fig.gca(projection='3d')  
for i, name in enumerate(data.target_names):
    ax.text3D(np.std(pca_result[:, 0][y==i])-i*500 ,np.std(pca_result[:, 1][y==i]),0,s=name, horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))  
# Plotting the PCA components    
ax.scatter(pca_result[:,0], pca_result[:, 1], c=y, cmap = plt.cm.Spectral,s=20, label=data.target_names)  
plt.show()  
