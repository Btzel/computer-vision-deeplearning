import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    
#import Pytorch
import torch
#we use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms

#are we using gpu ?
print("Gpu available:  {}".format(torch.cuda.is_available()))

if torch.cuda.is_available():
    device = 'cuda'
    print("device is cuda")
else:
    device = 'cpu'
    print("device is cpu")

#transform to a pytorch tensors and normalize our values between -1 and 1 
# (can be done for reduce in computational cost and avoid oscillations during training) 
#to ensure all features, or in our case, pixel intensities, are weighted equally when training our CNN
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

#define script directory
script_dir = os.path.dirname(os.path.abspath(__file__))


#load our training data and specify what transform to use when loading
trainset = torchvision.datasets.MNIST(script_dir + '/datasets/MNIST/train',
                                      train=True,
                                      download=True,
                                      transform=transform)
#load our test data and specify what transform to use when loading
testset = torchvision.datasets.MNIST(script_dir + '/datasets/MNIST/test',
                                      train=False,
                                      download=True,
                                      transform=transform)
#have 60000 train and 10000 test image samples for our training and test/validation processes
#each image have 28x28 pixel, grayscale images in mnist dataset
#print(trainset.data.shape) #([60000,28,28])
#print(testset.data.shape) #([10000,28,28])

#the first value in our dataset (not normalized yet)
#print(trainset.data[0])

#an image example
#image = trainset.data[0].numpy()
#display(image)

#prepare train and test loaders
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=0)

#we use the python function iter to return an iterator for our train_loader object
#dataiter = iter(trainloader)

#we use next to get the first batch of data from our iterator
#images,labels = dataiter._next_data()

#print(images.shape) #([128,1,28,28])
#print(labels.shape) #([128]) 

""" Building PyTorch CNN Model """

#nn is used as the basic building block for our network graphs
import torch.nn as nn
import torch.nn.functional as F

 #create model using python class
print("***STARTING TO BUILDING THE MODEL***")
class Net(nn.Module):
    def __init__(self):
        #super is a subclass of the nn.Module and inherits all its methods
        super(Net,self).__init__()
        
        #define layer objects
        #first cnn layer is conv layer using 32 filters of 3x3 kernel size, with stride of 1 and padding of 0
        self.conv1 = nn.Conv2d(1,32,3,1,0)
        #second cnn layer is conv layer using 64 filters of 3x3 kernel size, with stride of 1 and padding of 0
        self.conv2 = nn.Conv2d(32,64,3,1,0)
        #next layer is max pooling layer 2x2 with stride of 2
        self.pool = nn.MaxPool2d(2,2)
        #next layer is fully connected layer (called Linear in torch), takes output of max pooling
        #which is 12x12x64, and then connects it to a set of 128 nodes. the progress was:
        #28x28x1->3x3x1(conv2d)->26x26x32->3x3x32(conv2d)->24x24x64->2x2(maxpool2d(stride=2))->12x12x64
        #so we have 64 12x12 feature map to be used
        self.fc1 = nn.Linear(64*12*12,128)
        #next layer is fully connected layer connects the 128 nodes to 10 output nodes (which is our classes 0 to 9)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        #here we define forward propagation sequence
        #It's Conv1-Relu-Conv2-Relu-MaxPool-Flatten-FC1-FC2
        #remember relu is used to make negative numbers 0 in feature maps created after convolution filtering
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    #create an instance of the model and move it (memory and operations) to the cuda device
    net = Net()
    net.to(device=device)

    #print the model architecture
    print("Model built: " , net)

    ##import pytorch's optimization library
    import torch.optim as optim

    #we use cross entropy loss as our loss function
    criterion =  nn.CrossEntropyLoss()

    #for our gradient descent algorithm or optimizer
    #we use sthochastic gradient descent (SGD) with a learnin rate of 0.001
    #we set momentum to be 0.9
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    #training the model
    print("***TRAINING THE MODEL***")

    #in pytorch we use the building block functions to execute the training algorithm
    #get mini batch which consist of 128 inputs and their labels
    #initialize gradients with zero values
    #forward propagate and get outputs
    #use outputs to get loss
    #back propagate
    #update gradients using optimiser

    #so, loop over the training set multiple times (each time is called epoch)
    epochs = 10

    #empty arrays to store logs about what happened during training phase
    epoch_log = []
    loss_log = []
    accuracy_log = []

    #iterate for a specified number of epochs (which is 10 in this case)
    for epoch in range(epochs):
        print(f'starting epoch: {epoch+1}...')

        #keep adding or accumulating loss after each mini-batch(128 input) in running_loss
        running_loss = 0.0

        #iterate through trainloader iterator
        #each cycle is a minibatch
        for i,data in enumerate(trainloader,0):
            #get the inputs data is a list of [inputs,labels]
            inputs, labels = data

            #move data to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            #clear the gradients before training by setting to zero
            #required for a fresh start
            optimizer.zero_grad()

            #forward -> backprop + optimize
            outputs = net(inputs) #forward propagation
            loss = criterion(outputs,labels) #get loss (quantify the difference between the results and predictions)
            loss.backward() #back propagate to obtain new gradients for all nodes
            optimizer.step() #update the gradients(weights)

            #print training statistics - epoch/iterations/loss/accuracy
            running_loss += loss.item()

            if i % 50 == 49: #show loss every 50 mini batches
                correct = 0 #initialize variable to hold the count for the correct predictions
                total = 0 #initialize variable to hold the count of the number of labels iterated

                #dont need to gradients for validation, so wrap in
                #no_grad to save memory
                with torch.no_grad():
                    #iterate through the testLoader iterator
                    for data in testloader:
                        images,labels = data
                        #move data to gpu
                        images = images.to(device)
                        labels = labels.to(device)

                        #forward propagate test data batch through the model
                        outputs = net(images)

                        #get predictions from the maximum value of the predicted output tensor
                        #set dim = 1 as it specifies the number of dimensions to reduce
                        _, predicted = torch.max(outputs.data,dim=1)
                        #keep adding label size or length to the total variable
                        total += labels.size(0)
                        #keep a running total of the number of predictions predicted correctly
                        correct += (predicted == labels).sum().item()

                    accuracy = 100 * correct / total
                    epoch_num = epoch + 1
                    actual_loss = running_loss / 50
                    print(f'Epoch: {epoch_num}, Mini Batches completed: {(i+1)}, Loss: {actual_loss:.3f}, Test Accuracy: {accuracy:.3f}%')
                    running_loss = 0.0
        
        #store training stats after each epoch
        epoch_log.append(epoch_num)
        loss_log.append(actual_loss)
        accuracy_log.append(accuracy)

    print('***Training Finished***')
    #save the model
    model_path = script_dir + '/models/mnist_cnn_net.pth'
    torch.save(net.state_dict(),model_path)
    #load weights from the specified path
    net.load_state_dict(torch.load(model_path))
    #lets use forward propagate one mini batch and get the predicted outputs
    #use python iter to return an iterator for our train_loader object
    test_iter = iter(testloader)
    #use next to get the first batch of data from iterator
    images,labels = test_iter._next_data()
    #move data to gpu
    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images)

    #get the class predictions using torch.max
    _,predicted = torch.max(outputs,1)

    #print the batch predictions (128)
    print('predicted: ', ''.join('%1s' % predicted[j].cpu().numpy() for j in range (128)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.3f}%')

    from matplotlib import pyplot as plt
    #create a plot with secondary y axis with subplot
    fig,ax1 = plt.subplots()
    #set title and x axis label rotation
    plt.title("Accuracy & Loss vs Epoch")
    plt.xticks(rotation=45)

    #use twinx to create plot a secondary y axis
    ax2 = ax1.twinx()

    #create plot for loss_log and accuracy_log
    ax1.plot(epoch_log,loss_log,'g-')
    ax2.plot(epoch_log,accuracy_log,'b-')

    #set labels
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss',color='g')
    ax2.set_ylabel('Test Accuracy',color='b')

    plt.show()

