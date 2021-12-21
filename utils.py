
from skimage.transform import  rescale, radon, iradon
from skimage.morphology import disk
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch


def iradon_re(sinogram,theta):
    return iradon(sinogram,theta=theta, filter_name='hamming',interpolation='cubic')

def add_noise(SNR,sinogram):
    sinogram=np.array(sinogram)
    VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
    noise = np.random.randn(sinogram.shape[0],sinogram.shape[1])*np.sqrt(VARIANCE)
    return sinogram + noise

def compute_Laplacian(sinogram,n_neighbors = 10,precomputed=False):
    if not precomputed:
        data=sinogram.T
        A_nl= kneighbors_graph(data, n_neighbors=n_neighbors).toarray()
    else:
        A_nl=sinogram
    A_knn=0.5*(A_nl+A_nl.T)
    L = np.diag(A_knn.sum(axis=1)) - A_knn
    L=sparse.csr_matrix(L)
    eigenValues, eigenVectors=eigsh(L,k=3,which='SM')
    idx = np.argsort(eigenValues)
    return eigenVectors[:, idx[1]],eigenVectors[:, idx[2]]
def plot_L(e1,e2,theta_list):
    label=theta_list/360
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(e1, e2,c=label,cmap='hsv')
    plt.show()
def list_shift(theta,r):
    theta=np.array(theta)
    theta_len=len(theta)
    shift_len=int(theta_len*r)
    return np.concatenate((theta[shift_len:],theta[:shift_len]))
def computer_error(s1,s2):
    er=s1-s2
    return np.sqrt(np.mean(er**2))




class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx,:].float(), 'label': float(self.label[idx])}
        return sample


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)
    return res

def k_hop_subgraph(src, dst, A, A_csc, num_hops=2,directed=True):
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]
    return subgraph


def prepare_train_dataset(sinogram,n_neighbors,walk_len,batch_size,val_ratio,directed):
    A=kneighbors_graph(sinogram.T, n_neighbors=n_neighbors).toarray()
    A=A.astype(np.int)
    Awidth=A.shape[0]
    edge_index=np.arange(Awidth*Awidth)
    if directed:
        pos_edge_index=edge_index[A.reshape(-1)==1]
        matrix_index_mask=np.ones(A.shape,dtype=bool)
        matrix_index_mask[A==0]=False
        matrix_index_mask[np.eye(A.shape[0], dtype=int)==1]=True
        neg_edge_index=edge_index[matrix_index_mask.reshape(-1)==False]
        A=sparse.csr_matrix(A)
        A_csc=A.tocsc()

    else:
        A=A+A.T
        A[A>0]=1
        Atriu=np.triu(A,k=1)
        pos_edge_index=edge_index[Atriu.reshape(-1)==1]
        matrix_index_mask=np.ones(A.shape,dtype=bool)
        matrix_index_mask[A==0]=False
        matrix_index_mask[np.eye(A.shape[0], dtype=int)==1]=True
        matrix_index_mask= np.bitwise_not(matrix_index_mask)
        neg_edge_index=edge_index[np.triu(matrix_index_mask,k=1).reshape(-1)==True]
        A=sparse.csr_matrix(A)
        A_csc=None

    neg_edge_index=np.random.choice(neg_edge_index,len(pos_edge_index),replace=False)
    edge_index=np.concatenate((pos_edge_index,neg_edge_index),axis=0)
    train_data_list=np.empty((len(edge_index),2*walk_len),dtype=np.int64)
    train_label_list=np.empty(len(edge_index))

    index=0
    for i in tqdm(edge_index):
        xi=i//Awidth
        yi=i%Awidth
        A_t=k_hop_subgraph(xi,yi,A,A_csc,directed=directed)
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
        train_data_list[index]=wp.reshape(-1)
        train_label_list[index]=A_t[0,1]
        index=index+1

    train_data_list=torch.tensor(train_data_list,dtype=torch.int)
    train_label_list=torch.tensor(train_label_list,dtype=torch.int)

    perm = torch.randperm(train_data_list.size(0))
    train_data_list=train_data_list[perm]
    train_label_list=train_label_list[perm]

    num_div=int(val_ratio*train_data_list.size(0))
    train_data,val_data=train_data_list[num_div:],train_data_list[:num_div]
    train_label,val_label=train_label_list[num_div:],train_label_list[:num_div]
    train_loader = DataLoader(dataset=MyDataset(train_data,train_label), 
                                               batch_size=batch_size, 
                                               shuffle=True)
    val_loader = DataLoader(dataset=MyDataset(val_data,val_label), 
                                               batch_size=batch_size, 
                                               shuffle=True)

    return train_loader,val_loader


def prepare_test_dataset(sinogram_noise,A,n_neighbors,walk_len,batch_size,Awidth,iter_index,directed):
    edge_index=np.arange(Awidth*Awidth)
    if iter_index==-1:
        A=kneighbors_graph(sinogram_noise.T, n_neighbors=n_neighbors).toarray()
        A=A.astype(np.int)
        A_noise_d=kneighbors_graph(sinogram_noise.T, n_neighbors=n_neighbors*2).toarray()
        A_noise_d=A_noise_d.astype(np.int)
        if directed:
            A_noise_d=(A_noise_d-A)
            pos_edge_index=edge_index[A.reshape(-1)==1]
            neg_edge_index=edge_index[A_noise_d.reshape(-1)==1]
            A=sparse.csr_matrix(A)
            A_csc=A.tocsc()
        else:
            A=A+A.T
            A[A>0]=1
            A_noise_d=A_noise_d+A_noise_d.T
            A_noise_d[A_noise_d>0]=1
            A_noise_d=(A_noise_d-A)
            pos_edge_index=edge_index[np.triu(A,k=1).reshape(-1)==1]
            neg_edge_index=edge_index[np.triu(A_noise_d,k=1).reshape(-1)==1]
            A=sparse.csr_matrix(A)
            A_csc=None





    else:
        if iter_index==0:
            A_noise_d=kneighbors_graph(sinogram_noise.T, n_neighbors=n_neighbors).toarray()
            A_noise_d=A_noise_d.astype(np.int)
            if not directed:
                A_noise_d=A_noise_d+A_noise_d.T
                A_noise_d[A_noise_d>0]=1
        else:
            A_noise_dm=kneighbors_graph(sinogram_noise.T, n_neighbors=n_neighbors*iter_index).toarray()
            A_noise_d=kneighbors_graph(sinogram_noise.T, n_neighbors=n_neighbors*(iter_index+1)).toarray()
            A_noise_d=A_noise_d.astype(np.int)
            A_noise_dm=A_noise_dm.astype(np.int)
            if directed:
                A_noise_d=A_noise_d-A_noise_dm
            else:
                A_noise_d=A_noise_d+A_noise_d.T
                A_noise_d[A_noise_d>0]=1
                A_noise_dm=A_noise_dm+A_noise_dm.T
                A_noise_dm[A_noise_dm>0]=1
                A_noise_d=A_noise_d-A_noise_dm

        if directed:
            pos_edge_index=edge_index[A.reshape(-1)==1]
            neg_edge_index=edge_index[A_noise_d.reshape(-1)==1]
            A=sparse.csr_matrix(A)
            A_csc=A.tocsc()
        else:
            pos_edge_index=edge_index[np.triu(A,k=1).reshape(-1)==1]
            neg_edge_index=edge_index[np.triu(A_noise_d,k=1).reshape(-1)==1]
            A=sparse.csr_matrix(A)
            A_csc=None

    edge_index=np.concatenate((pos_edge_index,neg_edge_index),axis=0)
    edge_index=np.unique(edge_index)
    edge_index=edge_index.astype(np.int64)
    test_data_list=np.empty((len(edge_index),2*walk_len),dtype=np.int64)
    test_label_list=np.empty(len(edge_index))
    
    index=0
    for i in tqdm(edge_index):
        xi=i//Awidth
        yi=i%Awidth
        A_t=k_hop_subgraph(xi,yi,A,A_csc,directed=directed)
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

    return test_loader,edge_index



def update_A(edge_index,scores,Awidth,n_neighbors,directed):
    if not directed:
        edge_index_p=edge_index//Awidth
        edge_index_q=edge_index%Awidth
        edge_index_l=edge_index_p+edge_index_q*Awidth
        edge_index=np.concatenate((edge_index,edge_index_l),axis=0)
        scores=np.concatenate((scores,scores),axis=0)
    perm=np.argsort(edge_index)
    edge_index=edge_index[perm]
    scores=scores[perm]

    A_new=np.zeros((Awidth,Awidth),dtype=np.int64)
    index=0
    for i in range(Awidth):
        edge_node=[]
        score_node=[]
        while edge_index[index]<Awidth*(i+1):
            edge_node.append(edge_index[index]%Awidth)
            score_node.append(scores[index])
            index=index+1
            if index==len(edge_index):
                break

        edge_node=np.array(edge_node,dtype=np.int64)
        score_node=np.array(score_node)
        perm=np.argsort(score_node)
        edge_node=edge_node[perm[-n_neighbors:]]
        A_new[i,edge_node]=1

    if not directed:
        A_new=A_new+A_new.T
        A_new[A_new>0]=1
    return A_new



