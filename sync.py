from tqdm import tqdm
from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch.nn.functional as F
import networkx as nx
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
from skimage.morphology import disk
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np


from scipy import sparse
from scipy.sparse.linalg import eigsh
import sys
import plotly.graph_objects as go
sys.path.insert(0, '..')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch



from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from model import MLP
from utils import *

######train#####################
#########input coordinate  sinogram
############################
############################
############################
############################
#####################
n_neighbors=6
directed = False
#####################
walk_len=4
val_ratio=0.05
batch_size=256
lr=0.01
weight_decay=0.0005
#####################


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





####denoise
###########input distance matrix:sinogram_noise



import warnings
warnings.filterwarnings("ignore")


def updata_A_sycn(sinogram_noise,A,node_index,check_number,n_neighbors,walk_len,Awidth,directed):
    distance_i=sinogram_noise[node_index]
    check_list=np.argsort(distance_i)[:check_number]
    test_data_list=np.empty((len(check_list),2*walk_len),dtype=np.int64)
    test_label_list=np.empty(len(check_list))
    index=0
    for i in check_list:
        xi=node_index
        yi=i
        A_t=k_hop_subgraph(xi,yi,A,None,directed=directed)
        #A_t=A_t.tolil()
        A_tp=A_t.copy()
        A_tm=A_t.copy()
        A_tp[0,1]=1
        A_tm[0,1]=0
        wp=np.empty((walk_len,2))
        A_tp_t=A_tp.copy()
        A_tm_t=A_tm.copy()
        A_tp_t=A_tp@A_tp_t
        A_tm_t=A_tm@A_tm_t
        for j in range(walk_len):
            A_tp_t=A_tp@A_tp_t
            A_tm_t=A_tm@A_tm_t
            wp[j,0]=(A_tp_t[0,1])
            wp[j,1]=(A_tm_t[0,1])
        test_data_list[index]=wp.reshape(-1)
        test_label_list[index]=A_t[0,1]
        index=index+1
    test_data=torch.tensor(test_data_list,dtype=torch.int)
    test_label=torch.tensor(test_label_list,dtype=torch.int)
    test_loader = DataLoader(dataset=MyDataset(test_data,test_label), 
                                            batch_size=batch_size, 
                                            shuffle=False)
    scores,AUC=test(test_loader)
    argsort_score=np.argsort(-scores)
    argsort_index=check_list[argsort_score]
    #A=A.tolil()
    for i in argsort_index[:n_neighbors]:
        if i !=node_index:
            A[node_index,i]=1
            A[i,node_index]=1
    for i in argsort_index[n_neighbors:]:
        A[node_index,i]=0
        A[i,node_index]=0
    #A=A.tocsr()
    return A

#sinogram_noise=M
A=neighbotFromX(sinogram_noise.T, n_neighbors=n_neighbors)
Awidth=A.shape[0]
A=A.astype(np.int)
A=A+A.T
A[A>0]=1
Asparse=sparse.csr_matrix(A)
for i in tqdm(range(5000)):
    node_index=np.random.randint(Awidth)
    Asparse=updata_A_sycn(sinogram_noise,Asparse,node_index,check_number,n_neighbors,walk_len,Awidth,directed)

# for i in tqdm(range(Awidth)):
#     node_index=i
#     Asparse=updata_A_sycn(sinogram_noise,Asparse,node_index,check_number,n_neighbors,walk_len,Awidth,directed)
A=Asparse.todense()
A=np.array(A,dtype=np.int)
A=A+A.T
A[A>0]=1