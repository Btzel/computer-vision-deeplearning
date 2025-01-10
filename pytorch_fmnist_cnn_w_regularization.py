import PIL.Image
import os
import torch
import PIL
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#implementing data augmentation

data_transforms = {
    'train': transforms.Compose([
        #there are executed in order they called here
        #some of these transforms return a color image, hence why need to convert to grayscale
        transforms.RandomAffine(degrees=10,translate=(0.05,0.05),shear=5),
        transforms.ColorJitter(hue = .05,saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15,interpolation=InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((.5,),(.5,)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,),(.5,))
    ])
}

script_dir = os.path.dirname(os.path.abspath(__file__))

#load our training data and specify what transform to use when loading
trainset = torchvision.datasets.FashionMNIST(script_dir + '/datasets/FMNIST_reg/train',
                                      train=True,
                                      download=True,
                                      transform=data_transforms['train'])
#load our test data and specify what transform to use when loading
testset = torchvision.datasets.FashionMNIST(script_dir + '/datasets/FMNIST_reg/test',
                                      train=False,
                                      download=True,
                                      transform=data_transforms['val'])

#prepare train and test loaders
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=2)

#build model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        #add batchnorm, using 32 as input since we have 32 feature map
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3)
        #batchnorm again
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 12 * 12,128)
        self.fc2 = nn.Linear(128,10)
        #definin dropout function with a rate of .2
        # can apply this after any layer, but it's best suited after ReLU
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.dropout(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.dropout(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(x)
        x = x.view(-1,64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
net = Net()
net.to(device)

#addin l2 regularization with weight_decay in SGD, and many optimizer has l2 regularization parameter aswell
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.001)

epochs = 15 #need more epochs when use data augmentation,dropout, and l2 regularization

epoch_log = []
loss_log = []
accuracy_log = []

for epoch in range(epochs):
    print(f'Starting Epoch:  {epoch + 1}...')

    #keep adding or accumulation loss after each mini batch in running loss
    running_loss = 0.0

    #iterate through trainloader iterator
    #each cycle is a minibatch
    for i,data in enumerate(trainloader,0):
        #get the inputs
        inputs,labels = data

        #gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        #clear gradients before training by settin to zero for a fresh start
        optimizer.zero_grad()

        #forward -> backprop + optimize
        outputs = net(inputs)#fp
        loss = criterion(outputs,labels)# get loss
        loss.backward()#bp to obtain new gradients for all nodes
        optimizer.step()#update gradients

        #print trainin statistics
        running_loss += loss.item()
        if i % 100 == 99:
            correct = 0
            total = 0

            #we dont need gradients for validation so wrap in
            #no_grad to save memory
            with torch.no_grad():
                
                #iterate through the test loader
                for data in testloader:
                    
                    images,labels =data
                    #gpu
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = net(images)
                    _,predicted = torch.max(outputs.data,1)
                    #keep adding the label size or length to the total variable
                    total += labels.size(0)
                    #keep running total number of the predictions predicted correctly
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 100
                print(f'Epoch: {epoch_num}, Mini Batches Completed: {(i+1)}, Loss: {actual_loss:.3f}, Test Accuracy: {accuracy:.3f}%')
                running_loss = 0.0

    #store training stats after each epoch
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

correct = 0
total = 0

with torch.no_grad():
    test_counter = 0
    for data in testloader:
        images,labels = data
        test_counter += 1
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'test counter = {test_counter}')
accuracy = 100 * correct / total
print(f'accuracy of the network on the 10000 test images: {accuracy:.4f}%')

model_path = script_dir + '/models/fashion_mnist_cnn_net_reg.pth'
torch.save(net.state_dict(),model_path)

