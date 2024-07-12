import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    """ 1d CNNs followed by 3 fully connected layers """
    
    def __init__(self, num_class):
        super(NN,self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=8,kernel_size=13,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv1d(in_channels=8,out_channels=16,kernel_size=11,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=9,stride=1)
        self.dropout3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=7,stride=1)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(62976, 256)
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256,128)
        self.dropout6 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_class)
        
    def forward(self, x):
        
        x = F.max_pool1d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        
        x = F.max_pool1d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        
        x = F.max_pool1d(F.relu(self.conv3(x)),kernel_size=3)
        x = self.dropout3(x)
        
        x = F.max_pool1d(F.relu(self.conv4(x)),kernel_size=3)
        x = self.dropout4(x)
        
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2])))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x 
    
class NN2D(nn.Module):
    """ 2d CNNs followed by 4 fully connected layers """
    
    def __init__(self, num_class):
        super(NN2D,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(2064, 1024)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(1024, 256)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, num_class)
        
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2]*x.shape[3])))
        x = self.dropout3(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)

        x = F.relu(self.fc3(x))
        x = self.dropout5(x)
        
        x = self.fc4(x)
        
        return x 


class VGG16(nn.Module):
    """ VGG16 model """
    
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6144, 4096),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        
        out = self.layer1(x)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        out = self.layer4(out)
        
        out = self.layer5(out)
        
        out = self.layer6(out)
        
        out = self.layer7(out)
        
        out = self.layer8(out)
        
        out = self.layer9(out)
        
        out = self.layer10(out)
        
        out = self.layer11(out)
        
        out = self.layer12(out)
        
        out = self.layer13(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        
        out = self.fc1(out)
        
        out = self.fc2(out)
        
        return out
    