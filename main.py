from utils import *
from utils import *
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from model import MLP
import torch
#####################
n_neighbors=5
MIN_ANGLE=0
MAX_ANGLE = 360
SAMPLES = 512
SIGNAL_SIZE = 256
directed = False
SNR=10
#####################
walk_len=4
val_ratio=0.05
batch_size=256
lr=0.01
weight_decay=0.0005
n_neighbors=5
#####################




theta_list = np.linspace(MIN_ANGLE, MAX_ANGLE, SAMPLES)


parentPhantom = shepp_logan_phantom()
scale = SIGNAL_SIZE/ parentPhantom.shape[1]
parentPhantom = rescale(parentPhantom, scale=scale, mode='reflect')
#phantomDisk = disk(SIGNAL_SIZE/2)[1:,1:]
#parentPhantom=parentPhantom*phantomDisk
print('#Creating sinogram#')
sinogram =radon(parentPhantom, theta=theta_list,circle=True)



sinogram_noise=add_noise(SNR,sinogram)


print('#Plotting#')
print('##Computing Laplacian##')
e1,e2=compute_Laplacian(sinogram_noise,n_neighbors=n_neighbors)

theta_L=np.arctan2(e1, e2)
theta_L_sort=np.argsort(theta_L)
theta_recovered=theta_list[theta_L_sort]
print('##Iradoning##')
parentPhantom_recovered =iradon_re(sinogram_noise,theta_recovered)
parentPhantom_recovered_true_shift =iradon_re(sinogram_noise,theta_list)


imkwargs = dict(vmin=0, vmax=1)
fig, (ax0,ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 4.5),
                               sharex=False, sharey=False)
ax0.set_title('Ground Truth')
ax1.set_title('Reconstruction')
ax2.set_title('True Align')
ax3.set_title('Laplacian')

ax0.imshow(parentPhantom, cmap=plt.cm.Greys_r,**imkwargs)

ax1.imshow(parentPhantom_recovered, cmap=plt.cm.Greys_r,**imkwargs)
ax2.imshow(parentPhantom_recovered_true_shift, cmap=plt.cm.Greys_r)
ax3.scatter(e1, e2,c=theta_list/360,cmap='hsv')
plt.savefig('L.png')
plt.close()


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
    scores = torch.tensor([])
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

print('##reconnect##')
A=None
for i in range(-1):
    test_loader,edge_index=prepare_test_dataset(sinogram_noise,A,n_neighbors,walk_len,batch_size,SAMPLES,i,directed)
    scores,AUC=test(test_loader)
    A=update_A(edge_index,scores,SAMPLES,n_neighbors,directed)
    e1,e2=compute_Laplacian(A,n_neighbors = n_neighbors,precomputed=True)
    print('##Plotting##')
    theta_L=np.arctan2(e1, e2)
    theta_L_sort=np.argsort(theta_L)
    theta_recovered=theta_list[theta_L_sort]
    parentPhantom_recovered =iradon_re(sinogram_noise,theta_recovered)
    parentPhantom_recovered_true_shift =iradon_re(sinogram_noise,theta_list)


    imkwargs = dict(vmin=0, vmax=1)
    fig, (ax0,ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 4.5),
                                   sharex=False, sharey=False)
    ax0.set_title('Ground Truth')
    ax1.set_title('Reconstruction')
    ax2.set_title('True Align')
    ax3.set_title('Laplacian')

    ax0.imshow(parentPhantom, cmap=plt.cm.Greys_r,**imkwargs)

    ax1.imshow(parentPhantom_recovered, cmap=plt.cm.Greys_r,**imkwargs)
    ax2.imshow(parentPhantom_recovered_true_shift, cmap=plt.cm.Greys_r)
    ax3.scatter(e1, e2,c=theta_list/360,cmap='hsv')
    plt.savefig('L'+str(i)+'.png')
    plt.close()



