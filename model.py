import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size, stride,padding,downsample=False,downsample_stride=1):
        super().__init__()
        
        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.skip=nn.Sequential()
        if downsample==True:
          self.skip=nn.Sequential(
              nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=downsample_stride,bias=False),
              nn.BatchNorm2d(out_channels))
        
        else:
          self.skip=None
    def forward(self, x):

        identity=x

        out = self.block(x)
        if self.skip is not None:
          identity=self.skip(x)
        out+=identity
        out=F.relu(out)
        return out


class Model(nn.Module):
    def __init__ (self):
      super().__init__()
      self.layers = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(),
     
          SimpleResidualBlock(64,64,3,1,1),
          SimpleResidualBlock(64,64,3,1,1),
          SimpleResidualBlock(64,64,3,1,1),

          SimpleResidualBlock(64,128,3,2,1,True,2),
          SimpleResidualBlock(128,128,3,1,1),
          SimpleResidualBlock(128,128,3,1,1),
          SimpleResidualBlock(128,128,3,1,1),


          SimpleResidualBlock(128,256,3,2,1,True,2),
          SimpleResidualBlock(256,256,3,1,1),
          SimpleResidualBlock(256,256,3,1,1),
          SimpleResidualBlock(256,256,3,1,1),
          SimpleResidualBlock(256,256,3,1,1),
          SimpleResidualBlock(256,256,3,1,1),

          SimpleResidualBlock(256,512,3,2,1,True,2),
          SimpleResidualBlock(512,512,3,1,1),
          SimpleResidualBlock(512,512,3,1,1),
          nn.Dropout2d(0.3),
         

          nn.AdaptiveAvgPool2d(output_size=(1,1)),
          nn.Flatten(),
          
          )

      self.fc1=nn.Linear(512*1*1,4000)
      
      
      

    def forward(self,x):
          x=self.layers(x)       
          x = F.log_softmax(self.fc1(x),dim = 1)    
          return(x)

















