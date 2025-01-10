#load mnist dataset
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#print devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#preprocessing dataset
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

#gettin data in the right shape needed for keras
#need to add a 4th dimension to data thereby changing
#original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

#store the shape of a single image
input_shape = (img_rows,img_cols,1)

#change image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32') #uint8 originally

#normalize data by changing range from (0,255) to (0,1)
x_train /= 255.0
x_test /= 255.0

#one hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#count the number columns in hot encoded matrix
print("number of classes: " + str(y_test.shape[1]))
num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

"""***BUILDING THE MODEL***"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

#create model
model = Sequential()

#first conv layer using filter size 32 which reduces layer size to 26x26x32
#using relu to make negatives 0 in feature maps (filter outputs), and give input shape which is 28x28x1
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
#second conv layer using filter size 64 which reduces layer size to 24x24x64
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
#max pooling with kernel size of 2x2, reduces size to 12x12x64 with a minimal information loss
model.add(MaxPooling2D(pool_size=(2,2)))
#flatten tensor object before input into dense layer
#a flatten operation on a tensor reshapes the tensor to have the shape that is
#equal to the number of elements contained in tensor
#in this CNN its goes from 12x12x64 to 9216 * 1
model.add(Flatten())
#connect this layer to fully connected/dense layer of size 1*128
model.add(Dense(128,activation='relu'))
#create final fully connected/dense layer with an output for each class which is 10 in mnist (0-9)
#softmax is used for multiclass classification
model.add(Dense(num_classes,activation='softmax'))
#compile the model, this creates an object that stores the model created
#set optimizer to use SGD with learning rate of 0.001
#set loss function to be categorical cross entropy as its suitable for multiclass problem
#finally, the metrics to be accuracy
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.001),
              metrics = ['accuracy'])
#can use the summary function to display model layers and parameters
print(model.summary())

#define epochs and batch size
batch_size = 128
epochs = 25

#storing results here to plot later
#verbose = 1, sets training to output performance metrics every epoch
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose = 1,
                    validation_data = (x_test,y_test))
#obtain accuracy score using the evaluate function
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

#plotting training results
from matplotlib import pyplot as plt

#use history object
history_dict = history.history

#extract loss and validation losses
loss_values =history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
#get number of epochs and create array up to that number using range()
epochs = range(1,len(loss_values)+1)

#plot line charts for both validation and training loss
line1 =plt.plot(epochs,val_loss_values,label='Validation/Test Loss')
line2 =plt.plot(epochs,loss_values,label='Training Loss')
plt.setp(line1,linewidth=2.0,marker='+',markersize=10.0)
plt.setp(line2,linewidth=2.0,marker='+',markersize=10.0)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.show()

#plot line charts for both validation and training accuracy
line1 =plt.plot(epochs,val_acc_values,label='Validation/Test Accuracy')
line2 =plt.plot(epochs,acc_values,label='Training Accuracy')
plt.setp(line1,linewidth=2.0,marker='+',markersize=10.0)
plt.setp(line2,linewidth=2.0,marker='+',markersize=10.0)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend()
plt.show()

#save the model
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = script_dir + '/models/mnist_cnn_net.h5'
model.save(model_path)
print("Model Saved!...")

#import with load model
from tensorflow.keras.models import load_model
classifier = load_model(model_path)
(_,_),(_,y_test) = mnist.load_data()
#predicting all the test data
import numpy as np
pred = np.argmax(classifier.predict(x_test),axis=1)
print("Test Completed!...")

print(pred)
print(type(pred))
print(len(pred))
#use numpy to create an array that stores value of 1 when a misclassification occurs
result = np.absolute(y_test-pred)
misclassified_indices = np.nonzero(result>0)

#display the indices missclassified
print(f"indices of classified data are: \n{misclassified_indices}")

#create confusion matrix
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test,pred)
print(conf_mat)

#look at per class accuracy
class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

for (i,ca) in enumerate(class_accuracy):
    print(f'Accuracy for {i} : {ca:.3f}%')

#look at classification report
from sklearn.metrics import classification_report
class_rep = classification_report(y_test,pred)
print(class_rep)