import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

# -------------------------------------#
# perform training on CityScape dataset
# -------------------------------------#

'''
Notes for training on CityScape
   The weight files for the model will be saved in a folder named logs. 
   The saving operation will be performed after each epoch is completed.
   In case of interruption of training due to some reason, 
   you could specify the model_path in this train.py file (making it points to the most recent .pth file in logs folder) and resume training
'''


# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

def train():

    # this specifies whether to use Cuda to speed up training
    # for users with no Cuda GPUs, this can be change to False so that YOLO will use CPU to train
    Cuda = True

    # this is the path pointing to the cityscape_classes.txt
    classes_path = 'model_data/cityscape_classes.txt'

    # anchors_path points to the txt file that corresponds to anchors for YOLO 3
    # anchors_mask helps the algorithm to find anchors
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # this is where you specify path to your own .pth file when resuming training
    # default is the pre-trained YOLO weights
    model_path = 'model_data/yolo_weights.pth'

    # input size of the images, must be in multiples of 32
    input_shape = [416, 416]

    # First stage of training
    # We freeze the feature extraction network (DarkNet 53)
    # and only train on classification network
    # modify Freeze_epoch to specify how much epochs you wanna train
    Init_Epoch = 0
    Freeze_Epoch = 30
    Freeze_batch_size = 4
    Freeze_lr = 1e-3

    # Second stage of training
    # We unfreeze the feature extraction network (DarkNet 53)
    # This is when all the weights in the network will change
    # modify UnFreeze_epoch to specify how much epochs you wanna train
    # (number of epochs in unfreeze part = UnFreeze_Epoch - Freeze_Epoch)
    UnFreeze_Epoch = 60
    Unfreeze_batch_size = 2
    Unfreeze_lr = 1e-4

    # whether to freeze feature extraction network first, default to True
    Freeze_Train = True

    # whether to use multi thread to read data
    num_workers = 4

    # path to the two txt files we generated using voc_annotation.py
    train_annotation_path = 'cityscape_train.txt'
    val_annotation_path = 'cityscape_val.txt'

    # get classes and anchors
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    #create YOLO model, details please refer to file documentation
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    loss_list = []
    val_loss_list = []

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        print(train_lines)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            loss, val_loss = fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer,
                                           epoch,
                                           epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                                           Cuda)
            loss_list.append(loss)
            val_loss_list.append(val_loss_list)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small.")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            loss, val_loss = fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer,
                                           epoch,
                                           epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                                           Cuda)
            loss_list.append(loss)
            val_loss_list.append(val_loss)
            lr_scheduler.step()
    return loss_list, val_loss_list


if __name__ == "__main__":
    loss_list, val_loss_list = train()
    epoch_label = [i + 1 for i in range(len(loss_list))]

    # visualize the loss
    plt.plot(epoch_label, loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("loss.png")
    plt.show()
    plt.clf()
    # visualize the val_loss
    plt.plot(epoch_label, val_loss_list)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.savefig("val_loss.png")
    plt.show()
    plt.clf()
