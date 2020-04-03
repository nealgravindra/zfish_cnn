import tifffile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.optim as optim
from torch.utils import data


# train

total = time.time()

# initialize model
model = cnn()

# hyperparameters
n_epoch = 100
learning_rate = 0.001 # (default) lr=0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # change to AdaGrad

# save losses
train_losses = []
test_losses = []

for epoch in range(n_epoch):  # loop over the dataset multiple times
    train_loss_epoch = []
    test_loss_epoch = []

    start = time.time()
    running_loss = 0.0
    for i, d in enumerate(training_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = d

        # zero the parameter gradients
        model.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs.double(), labels.double())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # evaluate on test set
        test_loss = 0
        model.eval()
        with torch.no_grad() :
            dataiter = iter(validation_generator)
            inputs_val,labels_val = dataiter.next()
            outputs_val = model(inputs_val)
            batch_loss = criterion(outputs_val.double(),labels_val.double())
            test_loss += batch_loss.item()
        test_loss_epoch.append(test_loss)
        train_loss_epoch.append(running_loss)

        print('Epoch {:4d},batch {:3d}\ttrain_loss:{:.6f}\ttest_loss:{:.6f}'.format(epoch+1,i+1,running_loss,test_loss))
        running_loss = 0.0

    # average losses, add to overall train/test loss
    ave_train_loss=np.sum(train_loss_epoch)/len(train_loss_epoch)
    ave_test_loss=np.sum(test_loss_epoch)/len(test_loss_epoch)
    train_losses.append(ave_train_loss)
    test_losses.append(ave_test_loss)

    print('...Epoch {} in {:.2f}-s\t<train_loss>:{:.6f}\t<test_loss>:{:.6f}\n'.format(epoch+1,time.time() - start,ave_train_loss,ave_test_loss))

#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch, i + 1, running_loss / 2000))
#             running_loss = 0.0

print("\nTraining finished in {:.2f}-min".format((time.time() - total)/60))
