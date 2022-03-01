from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import torch
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys
import plotly.graph_objects as go
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import sys
import plotly.graph_objects as go
sys.path.insert(0, '..')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

import plotly.offline as pyo
import plotly.graph_objs as go
from icosphere import icosphere

def rotation3d(d3_points,phi,psi,theta):
    r = R.from_euler('xyz', [phi, psi, theta], degrees=True)
    return r.apply(d3_points)
def inplane_distance_points(dataset,M=100):
    print('search one')
    N=dataset.shape[0]
    data2d=torch.tensor(dataset,dtype=torch.float32)


    theta=torch.arange(0,2*np.pi,2*np.pi/M)
    rot_matrix=torch.empty(M,2,2)
    rot_matrix[:,0,0]=torch.cos(theta)
    rot_matrix[:,0,1]=-torch.sin(theta)
    rot_matrix[:,1,0]=torch.sin(theta)
    rot_matrix[:,1,1]=torch.cos(theta)

    distance_matrix=torch.empty((N,N))
    angle_matrix=torch.empty((N,N))
    for i in tqdm(range(N)):
        pts2dI=data2d[i,:,:]
        pts2dI_bt=torch.broadcast_to(pts2dI,(data2d.shape[0],M,data2d.shape[1],data2d.shape[2]))
        pts2d_rot=torch.einsum('nzp,mpq->nmzq', data2d, rot_matrix)
        diff=torch.norm(pts2d_rot-pts2dI_bt,dim=(2,3))
        diff,min_index=torch.min(diff,dim=1)
        distance_matrix[i,:]=diff
        angle_matrix[i]=theta[min_index]
    return distance_matrix.numpy(),angle_matrix.numpy()


def get_img(d3_points,resolution=50,xlim=[-1,1],ylim=[-1,1],theta=None,block=False):
    [xlim_l,xlim_u]=xlim
    [ylim_l,ylim_u]=ylim
    devx=(xlim_u-xlim_l)/resolution
    devy=(ylim_u-ylim_l)/resolution
    data_2d=d3_points[:,:2]
    if theta is not None:
        rot_matrix=np.empty((2,2))
        rot_matrix[0,0]=np.cos(theta)
        rot_matrix[0,1]=-np.sin(theta)
        rot_matrix[1,0]=np.sin(theta)
        rot_matrix[1,1]=np.cos(theta)
        data_2d=data_2d@rot_matrix
    data_2d=data_2d.T


    xcord=(data_2d[0,:]-(xlim_l))/devx
    ycord=(data_2d[1,:]-(ylim_l))/devy


    xcord[xcord>=resolution]=resolution-1
    xcord[ycord>=resolution]=resolution-1
    xcord[xcord<0]=0
    ycord[ycord<0]=0


    xcord=torch.tensor(xcord,dtype=torch.long)
    ycord=torch.tensor(ycord,dtype=torch.long)
    z=torch.ones_like(xcord,dtype=torch.int)
    img=torch.zeros((resolution,resolution),dtype=torch.int)
    img.index_put_((xcord, ycord), z, accumulate=True)
    img=img.numpy()
    if block:
        img[img>0]=1
    return img
def neighbotFromX(X,n_neighbors=10):
    A=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        diss_list=X[i]
        neb_list=np.argsort(diss_list)[1:n_neighbors+1]
        A[i][neb_list]=1
    return A
def plot_Laplacian(dataset,n_neighbors = 10,is_X=False,data_label=None,plot_line=False,A=None,use_A=False,location=None,use_loaction=False):
    color_RGBA=data_label
    if not use_A:
        if is_X==False:
            data=dataset.reshape(dataset.shape[0],-1)
            A_nl= kneighbors_graph(data, n_neighbors=n_neighbors).toarray()
        else:
            data=dataset.reshape(dataset.shape[0],-1)
            A_nl=neighbotFromX(data,n_neighbors=n_neighbors)
    else:
        A_nl=A
    # A_k=sparse.csr_matrix(A_nl)
    # A_knn = 0.5*(A_k + sparse.csr_matrix.transpose(A_k))
    # L = sparse.csr_matrix(sparse.diags(np.ones(A_nl.shape[0]))) - A_knn
    A_knn=0.5*(A_nl+A_nl.T)
    A_knn[A_knn>1]=1
    D=np.diag(A_knn.sum(axis=1))
    L = np.diag(A_knn.sum(axis=1)) - A_knn
    # D_invdiv2=np.diag(np.sqrt(1/A_knn.sum(axis=1)))
    # L=np.eye(A_knn.shape[0])-D_invdiv2@A_knn@D_invdiv2
    L=sparse.csr_matrix(L)
    eigenValues, eigenVectors=eigsh(L,k=4,which='SM')
    idx = np.argsort(eigenValues)
    fig = go.Figure()
    if use_loaction:
        datax,datay,dataz=location[:,0],location[:,1],location[:,2]
    else:
        datax,datay,dataz=eigenVectors[:, idx[1]],eigenVectors[:, idx[2]],eigenVectors[:, idx[3]]

    fig.add_trace(go.Scatter3d(
        x=datax,
        y=datay,
        z=dataz,
        mode='markers',
        marker=dict(
            size=5,               # set color to an array/list of desired values
            color=color_RGBA
        )
    ))

    # tight layout
    if plot_line:
        N=A_knn.shape[0]
        for i in tqdm(range(N)):
            xi=datax[i]
            yi=datay[i]
            zi=dataz[i]
            for j in range(N):
                if A_knn[i][j]>0:
                    xj=datax[j]
                    yj=datay[j]
                    zj=dataz[j]
                    c=(color_RGBA[i]+color_RGBA[j])/2
                    fig.add_trace(go.Scatter3d(x=[xi,xj],y=[yi,yj],z=[zi,zj],mode='lines',line=dict(width=3,color=c.reshape(1,4))))

    a=1.2


    camera = dict(
        eye=dict(x=1.25*a, y=1.25*a, z=1.25*a)
    )




    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene_camera=camera,showlegend=False)
    #fig.show()()
    return fig
def add_noise(SNR,sinogram):
    sinogram=np.array(sinogram)
    VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
    noise = np.random.randn(sinogram.shape[0],sinogram.shape[1],sinogram.shape[2])*np.sqrt(VARIANCE)
    return sinogram + noise





pts=np.load("bunny.npy")
pts=pts[np.random.permutation(pts.shape[0]),:]

pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2
pts_max=np.linalg.norm(pts,axis=1).max()
pts=pts/pts_max





nu = 10  # or any other integer
k=5
vertices, faces = icosphere(nu)

rotation_list=[]
pts_list=[]
pts_sub_list=[]
ax1,ax2,ax3=vertices[:,0],vertices[:,1],vertices[:,2]
r=np.sqrt(ax1**2+ax2**2)+1e-16
psi_list=np.arctan(ax3/r)/np.pi*180
phi_list=np.arctan2(ax2,ax1)/np.pi*180
phi_list=np.concatenate(([0],phi_list))
psi_list=np.concatenate(([0],psi_list))
data_number=len(psi_list)
print(data_number)
theta=0
for i in range(data_number):
    phi, psi=phi_list[i],psi_list[i]
    rotation_list.append(np.array([phi,psi,theta]))
    pts_t=rotation3d(pts,phi,psi,theta)
    pts_list.append(get_img(pts_t,resolution=30))
rotation_list=np.array(rotation_list)
pts_list=np.array(pts_list)




######generate inplane rotation imgs


pts_list_r=[]

theta_list=np.arange(0,360,k)
for i in tqdm(range(data_number)):
    pts_list_sub=[]
    img=pts_list[i]
    for theta in theta_list:
        pts_list_sub.append(rotate(img,theta,preserve_range=True).reshape(-1))
    pts_list_r.append(pts_list_sub)
pts_list_r=np.array(pts_list_r)

#######compute inplane rotation invariant  distance
D=np.zeros((data_number,data_number))
B=np.zeros((data_number,data_number))
for i in tqdm(range(data_number)):
    img=pts_list_r[i,0,:]
    img=np.broadcast_to(img,(pts_list_r.shape[1],pts_list_r.shape[2]))
    for j in range(data_number):
        img_j=pts_list_r[j,:,:]
        d=np.linalg.norm(img_j-img,axis=1)
        D[i,j]=np.min(d)
        B[i,j]=theta_list[np.argmin(d)]
###########




###########Plot projections
data_angle=rotation_list/180*np.pi
ax1=np.cos(data_angle[:,1])*np.cos(data_angle[:,0])
ax2=np.cos(data_angle[:,1])*np.sin(data_angle[:,0])
ax3=np.sin(data_angle[:,1])

color_RGBA=np.empty((data_number,4))

color_RGBA[:,0]=ax1
color_RGBA[:,1]=ax2
color_RGBA[:,2]=ax3
color_RGBA[:,3]=1
color_RGBA=color_RGBA*0.9+0.1


fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=ax1,
    y=ax2,
    z=ax3,
    mode='markers',
    marker=dict(
        size=5,               # set color to an array/list of desired values
        color=color_RGBA
    )
))

a=1.2
camera = dict(
    eye=dict(x=1.25*a, y=1.25*a, z=1.25*a)
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene_camera=camera,showlegend=False)
#fig.show()()
#fig.write_html("save/projection_location.html")
fig.write_image("fig1.pdf")





###########Plot true Laplacian
dataset=D
n_neighbors=5
data=dataset.reshape(dataset.shape[0],-1)
A_nl=neighbotFromX(data,n_neighbors=n_neighbors)
A_knn=0.5*(A_nl+A_nl.T)
A_knn[A_knn>1]=1
D=np.diag(A_knn.sum(axis=1))
L = np.diag(A_knn.sum(axis=1)) - A_knn

L=sparse.csr_matrix(L)
eigenValues, eigenVectors=eigsh(L,k=4,which='SM')
idx = np.argsort(eigenValues)
fig = go.Figure()
e1,e2,e3=eigenVectors[:, idx[1]],eigenVectors[:, idx[2]],eigenVectors[:, idx[3]]

fig = go.Figure()


e=np.array([e1,e2,e3]).T
###normlization
e=e/np.linalg.norm(e,axis=1,keepdims=True)

fig.add_trace(go.Scatter3d(
        x=e[:,0],
        y=e[:,1],
        z=e[:,2],
        mode='markers',
        marker=dict(
            size=5,               # set color to an array/list of desired values
            color=color_RGBA
        )
    ))

a=1.2
camera = dict(
    eye=dict(x=1.25*a, y=1.25*a, z=1.25*a)
)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene_camera=camera,showlegend=False)
#fig.show()()
fig.write_image("fig3.pdf")










######add noise
SNR=-2
pts_list_r_noise=add_noise(SNR,pts_list_r)
######Plot imgs
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,4))
ax2.imshow(pts_list_r_noise[0,0,:].reshape(30,30))
ax1.imshow(pts_list_r[0,0,:].reshape(30,30))
ax1.axis('off')
ax2.axis('off')
fig.savefig('fig4.pdf')



D_n=np.zeros((data_number,data_number))
B_n=np.zeros((data_number,data_number))
for i in tqdm(range(data_number)):
    img=pts_list_r_noise[i,0,:]
    img=np.broadcast_to(img,(pts_list_r_noise.shape[1],pts_list_r_noise.shape[2]))
    for j in range(data_number):
        img_j=pts_list_r_noise[j,:,:]
        d=np.linalg.norm(img_j-img,axis=1)
        D_n[i,j]=np.min(d)
        B_n[i,j]=theta_list[np.argmin(d)]




###########Plot noisy Laplacian
dataset=D_n
n_neighbors=5
data=dataset.reshape(dataset.shape[0],-1)
A_nl=neighbotFromX(data,n_neighbors=n_neighbors)
A_knn=0.5*(A_nl+A_nl.T)
A_knn[A_knn>1]=1
D=np.diag(A_knn.sum(axis=1))
L = np.diag(A_knn.sum(axis=1)) - A_knn
# D_invdiv2=np.diag(np.sqrt(1/A_knn.sum(axis=1)))
# L=np.eye(A_knn.shape[0])-D_invdiv2@A_knn@D_invdiv2
L=sparse.csr_matrix(L)
eigenValues, eigenVectors=eigsh(L,k=4,which='SM')
idx = np.argsort(eigenValues)
fig = go.Figure()
e1,e2,e3=eigenVectors[:, idx[1]],eigenVectors[:, idx[2]],eigenVectors[:, idx[3]]

fig = go.Figure()


e=np.array([e1,e2,e3]).T
e=e/np.linalg.norm(e,axis=1,keepdims=True)
e.shape

fig.add_trace(go.Scatter3d(
        x=e[:,0],
        y=e[:,1],
        z=e[:,2],
        mode='markers',
        marker=dict(
            size=5,               # set color to an array/list of desired values
            color=color_RGBA
        )
    ))
a=1.2


camera = dict(
    eye=dict(x=1.25*a, y=1.25*a, z=1.25*a)
)




fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene_camera=camera,showlegend=False)
#fig.show()()
fig.write_image("fig6.pdf")









######## WP

from utils import *
from sklearn.neighbors import kneighbors_graph
from model import MLP
import torch

######## True graph
data_angle=rotation_list/180*np.pi
ax1=np.cos(data_angle[:,1])*np.cos(data_angle[:,0])
ax2=np.cos(data_angle[:,1])*np.sin(data_angle[:,0])
ax3=np.sin(data_angle[:,1])

sinogram=np.array([ax1,ax2,ax3])


###train


#####################
n_neighbors=5
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
    for data in tqdm(loader):  # Iterate in batches over the training dataset.
        input = data['data'].to(device)
        label = data['label'].to(device)
        out = model(input)
        scores = torch.cat((scores,out),dim = 0)
        labels = torch.cat((labels,label.view(-1,1).cpu().clone().detach()),dim = 0)
    scores_np = scores.cpu().clone().detach().numpy()
    labels = labels.cpu().clone().detach().numpy()
    return scores_np.reshape(-1) ,roc_auc_score(labels, scores_np)
print('##training##')
for epoch in range(10):
    loss_epoch = train(train_loader,epoch)
    scores,AUC=test(val_loader)
    print('epoch:',epoch, ' AUC:',AUC)


A=None
SAMPLES=data_number
sinogram_noise=D_n


for i in [-1]:
    test_loader,edge_index=prepare_test_dataset_X(sinogram_noise,A,n_neighbors,walk_len,batch_size,SAMPLES,i,directed)
    scores,AUC=test(test_loader)
    A=update_A(edge_index,scores,SAMPLES,n_neighbors,directed)
    e1,e2,e3=compute_Laplacian(A,n_neighbors = n_neighbors,precomputed=True,dim=3)
    
for i in [0]:
    test_loader,edge_index=prepare_test_dataset_X(sinogram_noise,A,n_neighbors,walk_len,batch_size,SAMPLES,i,directed)
    scores,AUC=test(test_loader)
    A=update_A(edge_index,scores,SAMPLES,n_neighbors,directed)
    e1,e2,e3=compute_Laplacian(A,n_neighbors = n_neighbors,precomputed=True,dim=3)


for i in [0]:
    test_loader,edge_index=prepare_test_dataset_X(sinogram_noise,A,n_neighbors,walk_len,batch_size,SAMPLES,i,directed)
    scores,AUC=test(test_loader)
    A=update_A(edge_index,scores,SAMPLES,n_neighbors,directed)
    e1,e2,e3=compute_Laplacian(A,n_neighbors = n_neighbors,precomputed=True,dim=3)



fig = go.Figure()
e=np.array([e1,e2,e3]).T
e=e/np.linalg.norm(e,axis=1,keepdims=True)

fig.add_trace(go.Scatter3d(
        x=e[:,0],
        y=e[:,1],
        z=e[:,2],
        mode='markers',
        marker=dict(
            size=5,               # set color to an array/list of desired values
            color=color_RGBA
        )
    ))

a=1.2


camera = dict(
    eye=dict(x=1.25*a, y=1.25*a, z=1.25*a)
)




fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),scene_camera=camera,showlegend=False)
#fig.show()()

fig.write_image("fig8.pdf")