
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import sys
import plotly.graph_objects as go
sys.path.insert(0, '..')
from mpl_toolkits.mplot3d import Axes3D
from model import MLP
from utils import *
import os

def plot_Laplacian(dataset,n_neighbors = 10,is_X=False,data_label=None,pro=False,A=None,output_embedding=False,output_A=False,save_img=None):
    if A is None:
        if is_X==False:
            data=dataset.reshape(dataset.shape[0],-1)
            A_nl= kneighbors_graph(data, n_neighbors=n_neighbors).toarray()
        else:
            data=dataset.reshape(dataset.shape[0],-1)
            A_nl=neighbotFromX(data,n_neighbors=n_neighbors)
    else:
        A_nl=A
    A_knn=0.5*(A_nl+A_nl.T)
    if output_A:
        A_knn[A_knn>0]=1
        return A_knn 
    L = np.diag(A_knn.sum(axis=1)) - A_knn
    L=sparse.csr_matrix(L)
    eigenValues, eigenVectors=eigsh(L,k=4,which='SM')
    idx = np.argsort(eigenValues)
    x=eigenVectors[:, idx[1]]
    y=eigenVectors[:, idx[2]]
    z=eigenVectors[:, idx[3]]
    
    if pro:
        norm=np.sqrt(x**2+y**2+z**2)
        x=x/norm
        y=y/norm
        z=z/norm

    if output_embedding:
        return np.array([x,y,z]).T
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,               # set color to an array/list of desired values
            color=data_label
        )
    )])
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    if save_img is not None:
        fig.write_html(save_img)




data=np.load('data/Bunny/file_SNR-16_NProjs_4002.npz')
M=data['arr_0']
Ang=data['arr_1']
N=M.shape[0]
data_angle=Ang/180*np.pi
ax1=np.cos(data_angle[:,1])*np.cos(data_angle[:,0])
ax2=np.cos(data_angle[:,1])*np.sin(data_angle[:,0])
ax3=np.sin(data_angle[:,1])

color_RGBA=np.empty((N,4))

color_RGBA[:,0]=ax1
color_RGBA[:,1]=ax2
color_RGBA[:,2]=ax3
color_RGBA[:,3]=1
color_RGBA=color_RGBA*0.9+0.1

n_neighbors=4
plot_Laplacian(M,n_neighbors =n_neighbors,is_X=True,data_label=color_RGBA,pro=True,output_A=False)



directed = False
#####################
walk_len=4
val_ratio=0.05
batch_size=256
lr=0.01
weight_decay=0.0005
#####################

sinogram=np.array([ax1,ax2,ax3])### Training data


print('#Walkpooling#')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('##prepare_train_dataset##')
train_loader,val_loader=prepare_train_dataset(sinogram,n_neighbors,walk_len,batch_size,val_ratio,directed)


model=MLP(2*walk_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
criterion = torch.nn.BCEWithLogitsLoss()

def train(loader,epoch):
    model.train()
    loss_epoch=0
    for data in tqdm(loader):  # Iterate in batches over the training dataset.
        input = data['data'].to(device)
        label = data['label'].to(device)
        if input.shape[0]==1:
            continue
        out = model(input)
        loss = criterion(out.view(-1), label)  
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        loss_epoch=loss_epoch+loss.item()
    return loss_epoch/len(loader)

def test(loader):
    model.eval()
    loss_epoch=0
    scores = torch.tensor([]).to(device)
    labels = torch.tensor([])
    for data in loader:  # Iterate in batches over the training dataset.
        input = data['data'].to(device)
        label = data['label'].to(device)
        if input.shape[0]==1:
            continue
        out = model(input)
        scores = torch.cat((scores,out),dim = 0)
        labels = torch.cat((labels,label.view(-1,1).cpu().clone().detach()),dim = 0)
    scores_np = scores.cpu().clone().detach().numpy()
    labels = labels.cpu().clone().detach().numpy()
    #return scores_np.reshape(-1) ,roc_auc_score(labels, scores_np)
    return scores_np.reshape(-1), None
print('##training##')
for epoch in range(10):
    loss_epoch = train(train_loader,epoch)
    scores,AUC=test(val_loader)
    print('epoch:',epoch, ' AUC:',AUC)








import warnings
warnings.filterwarnings("ignore")




###test_________not sycn
# A=None
# SAMPLES=M.shape[0]
# sinogram_noise=M


# for i in [-1]:
#     test_loader,edge_index=prepare_test_dataset_X(sinogram_noise,A,n_neighbors,walk_len,batch_size,SAMPLES,i,directed)
#     scores,AUC=test(test_loader)
#     A=update_A(edge_index,scores,SAMPLES,n_neighbors,directed)


# plot_Laplacian(M,n_neighbors =n_neighbors,is_X=False,data_label=color_RGBA,pro=True,A=A,output_A=False)





###test_________sycn
sinogram_noise=M
check_number=2*n_neighbors
A=neighbotFromX(sinogram_noise.T, n_neighbors=n_neighbors)
N=A.shape[0]
A=A.astype(np.int)
A=A+A.T
A[A>0]=1
Asparse=sparse.csr_matrix(A)
for i in tqdm(range(int(N/2))):
    #node_index=i
    node_index=np.random.randint(N)
    Asparse=updata_A_sycn(sinogram_noise,Asparse,node_index,check_number,n_neighbors,walk_len,N,directed,test)
A=Asparse.todense()
A=np.array(A,dtype=np.int)
A=A+A.T
A[A>0]=1


plot_Laplacian(M,n_neighbors =n_neighbors,is_X=False,data_label=color_RGBA,pro=True,A=A,output_A=False)



