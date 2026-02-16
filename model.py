import torch 
import torch.nn as nn


#Dice loss function.
def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)

#----------------------------------------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.con1v1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.con1v1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x
    
#----------------------------------------------------------
class SkipConvConnection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        #You can use a convolutional block to ensure the match, not like my tinder, between the data dimensions.
        self.skip_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, size):
        skip = self.skip_conv(inputs)        
        out = self.bn(skip)
        out = self.relu(out)

        return out                              

#----------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.skip = SkipConvConnection(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv_block(x)                                           
        skip = self.skip(conv_out, size=(conv_out.shape[2], conv_out.shape[3])) 
        pooled = self.pool(conv_out)                                            
        return pooled, skip
    

class UpsampleBlock2D(nn.Module):
    def __init__(self, in_c,out_c):
        super().__init__()
        self.upsamlpe = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_c,out_c,kernel_size=3,stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU()

    def forward(self,inputs):
        up = self.upsamlpe(inputs)
        x = self.conv(up)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

#----------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = UpsampleBlock2D(in_channels, out_channels)
        self.conv_block = Conv2dBlock(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)                            
        x = torch.cat([x, skip], dim=1)                
        x = self.conv_block(x)                          
        return x


#----------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv_block(x)


#----------------------------------------------------------
class BasicallyUnet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        #ENCODER
        
        self.e1 = EncoderBlock(in_channels, base_channels)
        self.e2 = EncoderBlock(base_channels, base_channels*2)
        self.e3 = EncoderBlock(base_channels*2, base_channels*4)    
        self.e4 = EncoderBlock(base_channels*4, base_channels*8)
        
        #BOTTLENECK
        self.bottleneck = Bottleneck(base_channels*8, base_channels*16)  

        #DECODER  
        self.d1 = DecoderBlock(base_channels*16, base_channels*8, base_channels*8)  
        self.d2 = DecoderBlock(base_channels*8,  base_channels*4, base_channels*4)  
        self.d3 = DecoderBlock(base_channels*4,  base_channels*2, base_channels*2)  
        self.d4 = DecoderBlock(base_channels*2,  base_channels,   base_channels)  

        # Add final head
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()
  


    def forward(self, inputs):

        ##==========================================================
        #Run this code with a debugger(top right play figure, choose the derdernier option(debugInTerminalTitle)), place some breakpoints and check the shapes of the tensors at each step. 
        #This will help you understand, 4cheur, how the data flows through the network and how the dimensions change at each layer.
        #Use the debug console(bottom left), inspect variables, and step through the code to see how the tensors are transformed.
        #Like kevin's love life, you don't need to fully understand the layers operation fully. For now just the changes in the dimensions, and how the UNET treat the input and output.
        #You can do it! This are some commands to help you e1.shape, skip1.shape, x.shape.
        ##==========================================================
        
        # --- Encoder ---
        e1, skip1 = self.e1(inputs) # ---> inputs shape: (batch_size, in_channels, height, width) --> e1 shape: (batch_size, base_channels, height/2, width/2) --> skip1 shape: (batch_size, base_channels, height/2, width/2)
        e2, skip2 = self.e2(e1) # ---> e1 shape: (batch_size, base_channels, height/2, width/2) --> e2 shape: (batch_size, base_channels*2, height/4, width/4) --> skip2 shape: (batch_size, base_channels*2, height/4, width/4)
        e3, skip3 = self.e3(e2) # ---> e2 shape: (batch_size, base_channels*2, height/4, width/4) --> e3 shape: (batch_size, base_channels*4, height/8, width/8) --> skip3 shape: (batch_size, base_channels*4, height/8, width/8)
        e4, skip4 = self.e4(e3) # ---> e3 shape: (batch_size, base_channels*4, height/8, width/8) --> e4 shape: (batch_size, base_channels*8, height/16, width/16) --> skip4 shape: (batch_size, base_channels*8, height/16, width/16)

        # --- Bottleneck ---
        x = self.bottleneck(e4) # ---> e3 shape: (batch_size, base_channels*4, height/8, width/8) --> x shape: (batch_size, base_channels*8, height/8, width/8)
 
        # --- Decoder ---
        x = self.d1(x, skip4) # ---> x shape: (batch_size, base_channels*16, height/16, width/16) --> skip4 shape: (batch_size, base_channels*8, height/16, width/16) --> x shape after d1: (batch_size, base_channels*8, height/8, width/8)
        x = self.d2(x, skip3) # ---> x shape: (batch_size, base_channels*8, height/8, width/8) --> skip3 shape: (batch_size, base_channels*4, height/8, width/8) --> x shape after d2: (batch_size, base_channels*4, height/4, width/4)
        x = self.d3(x, skip2) # ---> x shape: (batch_size, base_channels*4, height/4, width/4) --> skip2 shape: (batch_size, base_channels*2, height/4, width/4) --> x shape after d3: (batch_size, base_channels*2, height/2, width/2)
        x = self.d4(x, skip1) # ---> x shape: (batch_size, base_channels*2, height/2, width/2) --> skip1 shape: (batch_size, base_channels, height/2, width/2) --> x shape after d4: (batch_size, base_channels, height, width)
        
        # --- Final head --- 
        x = self.final_conv(x) # ---> x shape: (batch_size, base_channels, height, width) --> x shape after final_conv: (batch_size, 1, height, width)
        x = self.sigmoid(x)      # ---> x shape: (batch_size, 1, height, width) --> x shape after sigmoid: (batch_size, 1, height, width)

        return x
        



#----------------------------------------------------------
if __name__ == "__main__":

    model = BasicallyUnet(in_channels=3, base_channels=64)
    dummy = torch.randn(2, 3, 512, 512)
    out   = model(dummy)
    print("Output shape:", out.shape)    


        
