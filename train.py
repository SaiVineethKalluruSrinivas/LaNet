import tables
import torch
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Module
from sklearn.utils import shuffle
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from utils import process_drive
from model import LaneNet
from tensorboardX import SummaryWriter


writer = SummaryWriter()
#Loading data
#input file path can be replaced with system arguments (sys.argv)
train_features_numpy_filename = "../data/TrainFeatures.h5"
train_targets_numpy_filename = "../data/TrainLabels.h5"
train_features_file = tables.open_file(train_features_numpy_filename, mode='r')
train_targets_file = tables.open_file(train_targets_numpy_filename, mode='r')

test_features_numpy_filename = "../data/TestFeatures.h5"
test_targets_numpy_filename = "../data/TestLabels.h5"
test_features_file = tables.open_file(test_features_numpy_filename, mode='r')
test_targets_file = tables.open_file(test_targets_numpy_filename, mode='r')

#Split train and test dataset
total_augmented_data_samples_sf = 23232
total_augmented_data_without_tw = 3872
total_test_set_size = 5000

index_list = np.arange(total_augmented_data_without_tw, total_augmented_data_samples_sf)
np.random.shuffle(index_list)
train_len = int(.95*len(index_list))
test_index_list = np.arange(total_test_set_size)

#function to get training and testing data
def get_data(drive_index, features=True, train=True):
    """
       fetches drive corresponding to data_index 
       if train=True, fetches from training dataset
       if train=False, fetches from testing dataset
    """
    if train:
        if features:
            return train_features_file.root.data[drive_index]
        else:
            return train_targets_file.root.data[drive_index]
    else:
        if features:
            return test_features_file.root.data[drive_index]
        else:
            return test_targets_file.root.data[drive_index]
        
#setting model_configs        
model_configs = {
    "l" : 800000,
    "s" : 50000,
    "d" : 50000,
    "m" : int(50000/2),
    "H" : 300,
    "lr" : 0.005,
    "num_epochs" : 4,
    "num_layers" : 2,
    "num_classes" : 2,
    "sampling_rate" : 2000,
    "batch_size" : 512
}

model_configs["n"] = int((model_configs["l"] - model_configs["d"])/model_configs["m"] + 1)
model_configs["D"] = int((model_configs["d"] - 500)/50 + 1)

#Initiating LaneNet model with model_configs
model = LaneNet(input_dim = model_configs["d"], 
                condensing_dim = model_configs["D"], # 991 for 50000
                seq_length = model_configs["n"],
                hidden_dim = model_configs["H"],
                batch_size = model_configs["batch_size"], 
                output_dim = model_configs["num_classes"], 
                num_layers = model_configs["num_layers"])

#load pretrained weights (if present)
#model = torch.load("LaneClassification800KSamples-50KSubSegmentLength.pt")

#Loss function as proposed in the paper
def WeightedLSTMCrossEntropyLoss(outputs, target):
    """
      Compute loss for outputs of model with respect to targets
      Outputs = O[i]
      Target = O*
    """
    total_vals = len(outputs)
    weights_per_cell = [(i+1)/total_vals for i in range(total_vals)]
    weights_per_cell = weights_per_cell/np.sum(weights_per_cell)
    loss = []
    for i in range(total_vals):
        this_loss = F.cross_entropy(outputs[i], target, weight=None, ignore_index=-100, reduction='mean')
        loss.append(weights_per_cell[i]*this_loss)
    return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

criterion = WeightedLSTMCrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr = model_configs["lr"])

# Training loop
for epoch in range(model_configs["num_epochs"]):  
    train_features_numpy_tr = (get_data(index) for index in index_list[:train_len])
    train_targets_numpy_tr = (get_data(index, features=False) for index in index_list[:train_len])
    drive_num = 0
    drive_in_batch_count = 0
    this_drive_batches = []
    all_labels = np.array([])
    for (drive, label) in zip(train_features_numpy_tr, train_targets_numpy_tr):
        processed_drive, num_batches, num_sub_batches = process_drive(drive = drive, 
                                                                      sub_drive_stride = model_configs["s"],
                                                                      sub_segment_stride = model_configs["m"],
                                                                      input_dim = model_configs["d"], 
                                                                      sampling_rate = model_configs["sampling_rate"],
                                                                      range_len = model_configs["l"])
        if (processed_drive is None):
            continue
        processed_drive_np = np.asarray(processed_drive)
        assert(model_configs["n"] == num_sub_batches)
        assert(num_batches == processed_drive_np.shape[0])
        assert(num_sub_batches == processed_drive_np.shape[1])
        #track the len of batch and collect drives = batch_size
        if (len(this_drive_batches) == 0):
            this_drive_batches = processed_drive
        
        else:
            this_drive_batches  += processed_drive
        all_labels = np.append(all_labels,  np.repeat(label, num_batches))
        drive_in_batch_count += num_batches            
        drive_num += 1
        if drive_in_batch_count < model_configs["batch_size"]:
            continue
            
        #train the batch and validate the model
        else:
            this_drive_batches_new = this_drive_batches[:model_configs["batch_size"]]
            all_labels_new = all_labels[:model_configs["batch_size"]]
            this_drive_batches_new, all_labels_new = shuffle(this_drive_batches_new, all_labels_new, random_state=3)
            features_curr_drive_batch_tensor = torch.from_numpy(np.array(this_drive_batches_new)).type(torch.FloatTensor)
            train = (Variable(features_curr_drive_batch_tensor))
            targets_curr_drive_batch_tensor = torch.from_numpy(all_labels_new).type(torch.LongTensor)
            targets = (Variable(targets_curr_drive_batch_tensor))
            optimizer.zero_grad()
            outputs = model(train)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            this_drive_batches = this_drive_batches[model_configs["batch_size"]:]
            all_labels = all_labels[model_configs["batch_size"]:]
            drive_in_batch_count -= model_configs["batch_size"]
            model.hidden = model.init_hidden()            
            
            accuracy = []
            
            #training accuracy
            for index, out in enumerate(outputs):
                predicted = torch.max(out.data, 1)[1].data.cpu().numpy()
                this_correct = np.count_nonzero(predicted == all_labels_new)
                accuracy.append(100.0*this_correct/float(model_configs["batch_size"]))
                writer.add_scalar("cell"+int(index), accuracy)
            
            print("Drives processed: {} \n Model loss: {} \n Accuracy: {}".format(drive_num, loss.data, accuracy))
            print("\n\n")

            
#save model
torch.save(model, "LaneClassification.pt")

#model validation
print("Validation accuracy")
with torch.no_grad():
    val_features_numpy_gen = (get_data(i) for index in index_list[train_len:])
    val_targets_numpy_gen = (get_data(i, features=False) for index in index_list[train_len:])
    correct = 0
    total = 0
    this_drive_batches_val = []
    all_labels_val = np.array([])
    val_drives_in_batch = 0
    drive_num = 0
    for val_drive, val_label in zip(val_features_numpy_gen, val_targets_numpy_gen):            
        processed_drive, num_batches, num_sub_batches = process_drive(drive = val_drive, 
                                                                      sub_drive_stride = model_configs["s"],
                                                                      sub_segment_stride = model_configs["m"],
                                                                      input_dim = model_configs["d"], 
                                                                      sampling_rate = model_configs["sampling_rate"],
                                                                      range_len = model_configs["l"])
        if (processed_drive is None):
            continue
        processed_drive_np = np.asarray(processed_drive)
        assert(model_configs["n"] == num_sub_batches)
        assert(num_batches == processed_drive_np.shape[0])
        assert(num_sub_batches == processed_drive_np.shape[1])
        if (len(this_drive_batches_val) == 0):
            this_drive_batches_val = processed_drive
        else:
            this_drive_batches_val  += processed_drive
        all_labels_val = np.append(all_labels_val,  np.repeat(val_label, num_batches))
        val_drives_in_batch += num_batches            
        drive_num += 1
        if val_drives_in_batch < model_configs["batch_size"]:
            continue
        else:
            this_drive_batches_new = this_drive_batches_val[:model_configs["batch_size"]]
            all_labels_new = all_labels_val[:model_configs["batch_size"]]
            features_curr_drive_batch_tensor = torch.from_numpy(np.array(this_drive_batches_new)).type(torch.FloatTensor)
            val = Variable(features_curr_drive_batch_tensor)
            targets_curr_drive_batch_tensor = torch.from_numpy(all_labels_new).type(torch.LongTensor)
            targets = Variable(targets_curr_drive_batch_tensor)
            outputs = model(val)[-1]

            predicted = torch.max(outputs.data, 1)[1].data.cpu().numpy()
            this_correct = np.count_nonzero(predicted == all_labels_new)
            accuracy = 100.0*this_correct/float(model_configs["batch_size"])
            print("Drives processed: {} \n Accuracy: {}".format(drive_num, accuracy))
            print("\n")
            correct += this_correct
            total += model_configs["batch_size"]

            this_drive_batches_val = this_drive_batches_val[model_configs["batch_size"]:]
            all_labels_val = all_labels_val[model_configs["batch_size"]:]
            val_drives_in_batch -= model_configs["batch_size"]

            model.hidden = model.init_hidden()
        accuracy = 100.0 * correct/float(total)
        print("Average Accuracy: {}".format(accuracy))
        print("\n\n")
        
#model testing
print("Testing accuracy")
with torch.no_grad():
    test_features_numpy_gen = (get_data(i) for index in test_index_list)
    test_targets_numpy_gen = (get_data(i, features=False) for index in test_index_list)
    correct = 0
    total = 0
    this_drive_batches_test = []
    all_labels_test = np.array([])
    test_drives_in_batch = 0
    drive_num = 0
    for test_drive, test_label in zip(test_features_numpy_gen, test_targets_numpy_gen):            
        processed_drive, num_batches, num_sub_batches = process_drive(drive = test_drive, 
                                                                      sub_drive_stride = model_configs["s"],
                                                                      sub_segment_stride = model_configs["m"],
                                                                      input_dim = model_configs["d"], 
                                                                      sampling_rate = model_configs["sampling_rate"],
                                                                      range_len = model_configs["l"])
        if (processed_drive is None):
            continue
        processed_drive_np = np.asarray(processed_drive)
        assert(model_configs["n"] == num_sub_batches)
        assert(num_batches == processed_drive_np.shape[0])
        assert(num_sub_batches == processed_drive_np.shape[1])
        if (len(this_drive_batches_test) == 0):
            this_drive_batches_test = processed_drive
        else:
            this_drive_batches_test  += processed_drive
        all_labels_test = np.append(all_labels_test,  np.repeat(test_label, num_batches))
        test_drives_in_batch += num_batches            
        drive_num += 1
        if test_drives_in_batch < model_configs["batch_size"]:
            continue
        else:
            this_drive_batches_new = this_drive_batches_test[:model_configs["batch_size"]]
            all_labels_new = all_labels_test[:model_configs["batch_size"]]
            features_curr_drive_batch_tensor = torch.from_numpy(np.array(this_drive_batches_new)).type(torch.FloatTensor)
            test = Variable(features_curr_drive_batch_tensor)
            targets_curr_drive_batch_tensor = torch.from_numpy(all_labels_new).type(torch.LongTensor)
            targets = Variable(targets_curr_drive_batch_tensor)
            outputs = model(test)[-1]

            predicted = torch.max(outputs.data, 1)[1].data.cpu().numpy()
            this_correct = np.count_nonzero(predicted == all_labels_new)
            accuracy = 100.0*this_correct/float(model_configs["batch_size"])
            print("Drives processed: {} \n Accuracy: {}".format(drive_num, accuracy))
            print("\n")
            correct += this_correct
            total += model_configs["batch_size"]

            this_drive_batches_test = this_drive_batches_test[model_configs["batch_size"]:]
            all_labels_test = all_labels_test[model_configs["batch_size"]:]
            test_drives_in_batch -= model_configs["batch_size"]

            model.hidden = model.init_hidden()
        accuracy = 100.0 * correct/float(total)
        print("Average Accuracy: {}".format(accuracy))
        print("\n\n")
        
        
# closing files
writer.close()
train_features_file.close()
train_targets_file.close()
