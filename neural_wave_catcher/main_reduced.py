import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from scipy.io import loadmat
from sklearn.metrics import roc_curve, auc



#  data normalization at the input

class Wave_Conv(nn.Module):
    def __init__(self, nT,Nch):
        super().__init__()
        
        
        # architecture - very simple convolution architecture
        
        
        N_kernels1 = 2#10
        N_kernels2 = 2
        size_kernel = 5
        size_pool = 5
        pool_stride = 2
        N_out1 = 50
        N_out2 = 20
        N_out3 = 10


        self.ln1 = nn.Linear(Nch, N_out1)

        self.conv1 = nn.Conv1d(N_out1,N_kernels1*N_out1,size_kernel, groups = N_out1)
        self.pool =nn.MaxPool1d(size_pool,pool_stride)

        self.ln2 = nn.Linear(N_out1*N_kernels1, N_out2)
        self.conv2 = nn.Conv1d(N_out2, N_kernels2*N_out2, size_kernel, groups = N_out2)

        #flatten

        #N_out_after_conv = 280#((N_out_after_conv-size_kernel+1)-size_pool+2)//pool_stride
        self.ln3 = nn.Linear(7,1) # CAALCULATE AUTOMATICALLY
        self.ln4 = nn.Linear(40, 1)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.ln1(x)
        x = torch.transpose(x, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))

        x = torch.transpose(x, 1, 2)
        x = self.ln2(x)
        x = torch.transpose(x, 1, 2)
        x = self.pool(F.relu(self.conv2(x)))


        x = torch.sigmoid(self.ln3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.sigmoid(self.ln4(x))
        return x


arr = loadmat('C:/Users/Fedosov/Downloads/arc.mat')
waves = arr['waves']

labels = arr['labels'][0]


#input = torch.randn(1, 1, 50)
nT = waves.shape[2]
Nch = waves.shape[1]
nB = waves.shape[0]//2
waves = np.array([waves[:nB,:,:],waves[nB:,:,:]])
wacon = Wave_Conv(nT,Nch)

#out = wacon(input)
#print(out)


criterion = nn.MSELoss()
wacon.zero_grad()     # zeroes the gradient buffers of all parameters
# create your optimizer
optimizer = optim.Adam(wacon.parameters(), lr=0.005)

Nsteps = 1000
history = np.zeros(Nsteps)



train_p = 0.8

nB_train = int(nB*train_p)
size_Batch = 800


individual_norm = np.sqrt(np.sum(np.sum(waves[:,:,:,:]**2,axis = 3),axis = 2)[:,:,np.newaxis,np.newaxis])/(nT*Nch)

waves = waves/individual_norm
'''
for i in range(Nsteps):

    idc = np.random.randint(0,nB_train,size_Batch)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers

    Batch = torch.from_numpy(np.concatenate(waves[:,idc,:]))
    lbl = torch.from_numpy(np.concatenate([np.ones(size_Batch),np.zeros(size_Batch)])[:,np.newaxis])
    output = wacon(Batch.float())
    loss = criterion(output, lbl.float())
    print(loss.detach().numpy())
    history[i] = loss.detach().numpy()
    loss.backward()
    optimizer.step()

torch.save(wacon, 'C:/Users/Fedosov/Downloads/model_1')


plt.figure()
plt.plot(history)
'''

wacon = torch.load('C:/Users/Fedosov/Downloads/model_1')

test_waves = np.concatenate(waves[:,nB_train:,:,:])
test_labels = np.concatenate(np.array([np.ones(1000),np.zeros(1000)]))

#results = np.zeros(nB-nB_train)
Batch = torch.from_numpy(test_waves)

results = wacon(Batch.float()).detach().numpy()


fpr_tpr = roc_curve(test_labels, results)

my_auc = auc(fpr_tpr[0],fpr_tpr[1])
my_auc2 = auc(fpr_tpr[1],fpr_tpr[0])
print(my_auc)
print(my_auc2)
plt.figure()
plt.plot(waves[0,20,:,:].T)
plt.figure()
plt.plot(waves[1,20,:,:].T)

plt.figure()
plt.plot(fpr_tpr[0],fpr_tpr[1])
plt.show()