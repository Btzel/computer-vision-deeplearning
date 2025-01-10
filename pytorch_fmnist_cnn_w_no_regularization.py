#import pytorch
import torch
import PIL

#we use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms

#using gpu ?
if torch.cuda.is_available():
    device = 'cuda'
    print("gpu is avaiblable")
else:
    device = 'cpu'
    print("there is no gpu available")

#tranform to a pytorch tensors and the normalize our values between -1 and 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

#define script directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

#load our training data and specify what transform to use when loading
trainset = torchvision.datasets.FashionMNIST(script_dir + '/datasets/FMNIST/train',
                                      train=True,
                                      download=True,
                                      transform=transform)
#load our test data and specify what transform to use when loading
testset = torchvision.datasets.FashionMNIST(script_dir + '/datasets/FMNIST/test',
                                      train=False,
                                      download=True,
                                      transform=transform)

#prepare train and test loaders
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=2)

#create a list with our class names
classes = ('T-shirt/top','Trouser','Pullover','Dress',
           'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')


#function to show an image
from matplotlib import pyplot as plt
import numpy as np
def display(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#get some random training images
dataiter = iter(trainloader)
images,labels = dataiter._next_data()

#show images
display(torchvision.utils.make_grid(images))

#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

#building and training the model with no regularization
#import pytorchs optimization library and nn
#nn is used as the basic building block for our network graphs
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

net = Net()
net.to(device)

#use cross entropy as loss function
criterion = nn.CrossEntropyLoss()

#sgd for gradient descent with lr=0.001, momentum=0.9
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

epochs = 5

#log arrays
epoch_log = []
loss_log = []
accuracy_log = []

#iterate for a specified epochs
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
                actual_loss = running_loss / 50
                print(f'Epoch: {epoch_num}, Mini Batches Completed: {(i+1)}, Loss: {actual_loss:.3f}, Test Accuracy: {accuracy:.3f}%')
                running_loss = 0.0

    #store training stats after each epoch
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print("training is finished")

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

model_path = script_dir + '/models/fashion_mnist_cnn_net.pth'
torch.save(net.state_dict(),model_path)