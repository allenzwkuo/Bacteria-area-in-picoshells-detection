import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# defines double convolution block 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        # defines two convolution layers with each followed by batch normalization and ReLU activation
        self.conv = nn.Sequential(
            # conv layer 1
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLU activation
            # conv layer 2
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU(inplace=True), # ReLU activation
        )

    # forward pass for double conv block
    def forward(self, x):
        return self.conv(x)
    
# defines UNET model
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder (downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # decoder (upsampling) 
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # bottle neck where image is smallest
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # final conv layer outputting desired channels (1 in this case bc black and white)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # stores skipped connections for upsampling concatenation
        skip_connections = []

        # downsampling
        for down in self.downs:
            x = down(x) # pass through double conv
            skip_connections.append(x) # save feature map for skip connections
            x = self.pool(x) # downsample with max pooling

        # bottleneck of UNET
        x = self.bottleneck(x)

        # reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # transposed convolution for upsampling
            skip_connection = skip_connections[idx//2] # access corresponding skip connection feature map 

            # check to make sure spatial dimensions match
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])        

            # concatenate skip connection with currect output
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # pass concatenated tensor through double conv
            x = self.ups[idx+1](concat_skip)

        # final convolution for output
        return self.final_conv(x)