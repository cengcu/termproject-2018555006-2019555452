import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("veriler.txt", sep="\t")
# data.to_csv('dataset.csv', mode='w', index=False)

def kernel_function(X, X_i, h):
    power = np.sqrt(np.square(X-X_i).sum())
    den = 2*h*h
    return np.square((0.399/h))*np.exp(-(power/den))
def gradient_ascent(new_feature, h):
    for i in range(10):
        new_values = [0, 0]
        sum_k = 0
        for j in range(data.shape[0]):
            k = kernel_function(new_feature, data.iloc[j][['X','Y']].values, h)
            sum_k += k
            new_values += data.iloc[j][['X','Y']].values*k
        new_feature = new_values/sum_k
    return np.round(new_feature,2)
def Denclue(data, h, t):
    data['height'] = 0.000
    data['new_F1'] = 0.000
    data['new_F2'] = 0.000
    no = 20
    for i in range(data.shape[0]):
        if no==i+1:
            print('*', end="")
            no+=20
        feat_1 = 0.0
        new_Feature = [0, 0]
        for j in range(data.shape[0]):
            k = kernel_function(data.iloc[i][['X','Y']].values, data.iloc[j][['X','Y']].values, h)
            feat_1 += k
            new_Feature += data.iloc[j][['X','Y']].values*k
        data['height'][i] = (feat_1/len(data)-t)
        new_Feature /= feat_1
        if data['height'][i]<0:
            data['height'][i] = 0
        else:
            g = gradient_ascent(new_Feature, h)
            data['new_F1'][i] = g[0]
            data['new_F2'][i] = g[1]
    centers= pd.DataFrame({
        'F1_center': data[data['new_F1']!=0]['new_F1'].unique(),
        'F2_center': data[data['new_F2']!=0]['new_F2'].unique(),
        'Cluster': np.arange(1, len(data[data['new_F1']!=0]['new_F1'].unique())+1)
    })
    print('\n\nCluster Centers')
    print(centers)
    data['Cluster'] = -1
    for i in range(data.shape[0]):
        for j in range(centers.shape[0]):
            if data.iloc[i]['new_F1'] == centers.iloc[j]['F1_center'] and data.iloc[i]['new_F2'] == centers.iloc[j]['F2_center']:
                data['Cluster'][i] = centers.iloc[j]['Cluster']
                break
    print('\nNo of Datapoints in Each Cluster')
    print(data.groupby('Cluster')['Cluster'].count())
h = 0.3
t = 0.01
Denclue(data, h, t)
plt.figure(figsize=(8, 8))
sns.scatterplot(data=data, x='X', y='Y', hue='Sınıf', palette="deep", s=130)
plt.grid(True)
plt.show()