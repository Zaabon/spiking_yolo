# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from SpikingNN.spiking_model import*
#from automotive_dataset.data_loader import Prophesee
import numpy as np

from automotive_dataset.src.io.box_loading import reformat_boxes
from automotive_dataset.src.io.psee_loader import PSEELoader
import train as yolo

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'spiking_model'
data_path = './raw/'  # todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Running on", device, "\n")

"""
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
"""

# Data
path = "/home/z/Downloads/ATIS Automotive Detection Dataset/"
video = ["17-03-30_12-53-58_183500000_243500000_td.dat"]
video = PSEELoader(path + video[0])
bb = np.load("/home/z/Downloads/ATIS Automotive Detection Dataset/17-03-30_12-53-58_183500000_243500000_bbox.npy")
#bb = reformat_boxes(bb)  # Flips two last elements for some reason
print(bb)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

# Init SNN
snn = SCNN()
#snn.load_state_dict(torch.load('./checkpoint/ckpt' + names + '.t7'))
snn.eval()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

# Init YOLO
yolo = yolo.Train()


for epoch in range(num_epochs):                                     # - EPOCH -
    print('Epoch [%d/%d]' % (epoch+1, num_epochs))
    # Train
    running_loss = 0  # Only used for printing
    start_time = time.time()

    bb_tracker = 99999
    bb_index = 0
    outputs = torch.zeros((255, 3, 1, 255), device=device)
    while not video.done:
        # Perform prediction every ls time step
        if video.current_time >= bb_tracker:  # Check outputs every ls micro seconds
            labels = []  # this will need to be formatted for yolo
            while bb[bb_index][0] == bb_tracker:
                labels.append(bb[bb_index])
                bb_index += 1

            # Call YOLO and receive loss or predictions
            loss = yolo.predict(outputs, labels, learning_rate)  # Should have separate learning rate for yolo
            # Compare predictions with labels
            # Backpropagate yolo based on loss <- yolo does this by itself
            # Propagate through the intermediate layer
            #   This should reset ctx
            loss.backward()   # Calculate gradients <- You will need to set the loss to the SNN before you can calculate gradients with backward()
            optimizer.step()  # Update weights

            print(outputs)
            outputs = torch.zeros(outputs.size(), device=device)  # Reset outputs
            bb_tracker += ls


        snn.zero_grad()
        optimizer.zero_grad()

        data = video.load_delta_t(ts)  # Group data

        image = torch.zeros([1, 1, 304, 240])

        for i, event in enumerate(data):  # This reformatting takes 75% of running time
            image[0][0][event[2]][event[3]] += 1  # Tensors have a lot of overhead so should use one larger tensor
        image = image / max(image)  # Normalize firing probability to 1

        #data = data_loader.load_batch()
        #if data is None:
        #    break
        #(events, bb) = data

        #print(bb)
        images = image.float().to(device)
        outputs += snn(images)
        del images  # Free up GPU memory

        #print("Batch [%.2f%%] %.1fs" % (video.current_time/6000000, time.time()-start_time))
        print("Progress [%.2fs/60s] Real time %.1fs" % (video.current_time / 1000000, time.time() - start_time))


        #import torch
        #import gc

        #for obj in gc.get_objects():
        #    try:
        #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #            print(type(obj), obj.size())
        #    except:
        #        pass

        """
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1) % batch_size == 0:
            print('Batch [%d/%d]\t Loss: %.5f\t Time: %.0fs'
                  % (i+1, len(train_dataset)//batch_size, running_loss, time.time()-start_time))
            running_loss = 0
        """
    """
    # Test
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):  # - TEST BATCHES -
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)  # Loss is never used in the test phrase, can probably be deleted without side effects
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
    acc = 100. * float(correct) / float(total)
    print('Test Accuracy of the model on the 10000 test images: %.3f' % acc, "\n\n")
    acc_record.append(acc)
    """
    """
    # Save
    if epoch % 5 == 0 and epoch != 0:
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc
    """

