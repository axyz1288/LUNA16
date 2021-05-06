#!/usr/bin/env python
# coding: utf-8

# # Setting argument

# In[1]:


import argparse

# setting hyperparameters
parser = argparse.ArgumentParser(description='net')
parser.add_argument('--batchsize', '-b', 
                    type=int, 
                    default=32,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', 
                    type=int, 
                    default=200,
                    help='Number of epoch over the dataset to train')
parser.add_argument('--iter', '-i', 
                    type=int, 
                    default=5,
                    help='Number of iteration over the dataset to train')
parser.add_argument('--use-adasum', 
                    action='store_true', 
                    default=False,
                    help='use adasum algorithm to do reduction')
# create arg object
args = parser.parse_args([])

# setting state
CLASS_NUM = 2


# # Horovod

# In[2]:


import horovod.torch as hvd
import torch

hvd.init()
torch.cuda.manual_seed(42)


# # Load Data of LUNA16

# In[3]:


from Luna16 import Luna16
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainset = Luna16(csv_file='../LUNA16/CSVFILES/fake.csv',
                  root_dir='../LUNA16/data',
                  transform=transform,
                  is_segment=True,
                  seg_size=[14,29,29])

train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, 
                                                                num_replicas=hvd.size(), 
                                                                rank=hvd.rank())

torch.set_num_threads(1)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=args.batchsize, 
                                          sampler=train_sampler,
                                          num_workers=1,
                                          pin_memory=True)


# # To GPU

# In[4]:


# Assuming that we are on a CUDA machine, this should print a CUDA device:
torch.cuda.set_device(hvd.local_rank())
torch.cuda.empty_cache()


# # Build net

# In[5]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(1, 64, kernel_size=(3,3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.BatchNorm3d(64))
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Conv3d(128, 128, kernel_size=(1,1,1), padding=0),
                        nn.LeakyReLU(),
                        nn.BatchNorm3d(128),
                        nn.Conv3d(128, 64, kernel_size=(3,3,3), padding=1),
                        nn.LeakyReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv3d(64, 32, kernel_size=(3,3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.BatchNorm3d(32))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(32, CLASS_NUM, kernel_size=(3,3,3), padding=1),
                        nn.LeakyReLU())

        
    def forward(self, x):
        out = self.layer1(x)
        tmp = out
        out = self.layer2(out)
        out += tmp
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
net = Net().cuda()


# In[6]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),
                       lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


# # Horovod optimizer

# In[7]:


# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=compression)
                                     #op=hvd.Adasum if args.use_adasum else hvd.Average)


# # Average

# In[8]:


def metric_average(val, name):
    tensor = torch.tensor(val, dtype=torch.float)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# In[]:

if hvd.rank() == 0:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()



# In[9]:


import Accuracy
import time
s_time = time.time()
threshold = 0.7
print('[epoch,  iter]')
for epoch in range(args.epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    for i, data in enumerate(trainloader):
        # [data, label][batch][channel][depth][h][w]
        inputs, labels = data['data'].cuda(), data['label'].cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if(i % args.iter == args.iter-1):    # print every arg.batchsize mini-batches
            print('[%5d, %5d] loss: %.3f' %
                  (epoch+1, i+1, running_loss / args.iter))
            if hvd.rank() == 0:
                writer.add_scalar('Loss', running_loss / args.iter, epoch*args.iter + (i+1)/args.iter)
            running_loss = 0.0
        
        # Accuracy
        for batch_idx in range(outputs.shape[0]):
            label = labels[batch_idx].cpu()
            prediction = F.softmax(outputs[batch_idx], dim=0).cpu()
            tp, fp, tn, fn = Accuracy.eval(prediction, label, threshold)
            TP += tp; FP += fp; TN += tn; FN += fn

    sensitivity = TP/(TP+FN) if (TP+FN)!=0 else 0
    precision = TP/(TP+FP) if (TP+FP)!=0 else 0
    fpr = FP/(FP+TN) if (FP+TN)!=0 else 0
    sensitivity = metric_average(sensitivity, 'Sensitivity')
    precision = metric_average(precision, 'Precision')
    fpr = metric_average(fpr, 'False Positive Rate')
    if hvd.rank() == 0:
        print('-----------------------------------------------')
        print('Sensitivity: {:3.2%}'.format(sensitivity))
        print('Precision: {:3.2%}'.format(precision))
        print('False Positive Rate: {:3.2%}'.format(fpr))
        print('-----------------------------------------------')
        writer.add_scalar('Sensitivity', sensitivity*100, epoch)
        writer.add_scalar('Precision', precision*100, epoch)
        writer.add_scalar('False Positive Rate', fpr*100, epoch)
        if(epoch % 50 == 50-1):
            torch.save(net.state_dict(), 'net_parameters.pkl')

print('Total time: {}s'.format(time.time()-s_time))
print('Finished Training')

# scp -P 12345 net.py root@192.168.10.118:/home/horovod/luna16
# horovodrun -np 5 -H 127.0.0.1:2,192.168.10.119:2,192.168.10.118:1 --verbose --start-timeout 300 python3 net.py