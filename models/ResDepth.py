import torch  
import torch.nn as nn  
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, encoder_pretrained=False):
        super(Encoder, self).__init__()
        self.resnet = models.resnet101(pretrained=encoder_pretrained)
        
        self.stage_1 = nn.Sequential(self.resnet.conv1,
                                    self.resnet.bn1,
                                    self.resnet.relu
                                    )

        self.stage_2 = nn.Sequential(self.resnet.maxpool,
                                    self.resnet.layer1,
                                    )
        
        self.stage_3 = self.resnet.layer2
        self.stage_4 = self.resnet.layer3
        self.stage_5 = self.resnet.layer4
    
    def forward(self, x):
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)
        x5 = self.stage_5(x4)

        out = [x1, x2, x3, x4, x5]

        return out

class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upsample, self).__init__() 

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.convA = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_channels, output_channels, 3, 1, 1)

    def forward(self, x, concat_with):
        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)

        # print(upsampled_x.shape)
        
        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x


class Decoder(nn.Module):
    def __init__(self, max_depth, num_features=2048, decoder_width=0.5, scales=[1, 2, 4, 8]):
        super(Decoder, self).__init__()

        self.max_depth = max_depth
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, 1, 1, 1)

        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1536, 512)
        self.upsample3 = Upsample(768, 256)
        self.upsample4 = Upsample(320, 64)

        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x_block0 = features[0]
        x_block1 = features[1]
        x_block2 = features[2]
        x_block3 = features[3]
        x_block4 = features[4]

        x0 = self.conv2(x_block4)
        x1 = self.upsample1(x0, x_block3)
        x2 = self.upsample2(x1, x_block2)
        x3 = self.upsample3(x2, x_block1)
        x4 = self.upsample4(x3, x_block0)
        out = self.conv3(x4)

        out = self.sigmoid(out) * self.max_depth

        return x0, x1, x2, x3, x4, out

class DOG_moddule(nn.Module):
    def __init__(self, max_depth):
        super(DOG_moddule, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth
    
    def forward(self, feature):
        C = feature.shape[1]
        X = self.sigmoid(feature)
        g2 = torch.sum(X * X, dim=1).unsqueeze(1) / C
        Xs = F.interpolate(g2, scale_factor=0.5, mode='bilinear')
        Xb = F.interpolate(Xs, scale_factor=2, mode='bilinear')
        DOG = torch.abs(g2 - Xb) * self.max_depth

        return DOG

class ResDepth(nn.Module):
    def __init__(self, max_depth=80, encoder_pretrained=True):
        super(ResDepth, self).__init__()

        self.encoder = Encoder(encoder_pretrained=encoder_pretrained)
        self.decoder = Decoder(max_depth=max_depth)
        self.DOG = DOG_moddule(max_depth=max_depth)
    
    def forward(self, x):
        x = self.encoder(x)
        x0, x1, x2, x3, x4, out = self.decoder(x)

        x_2 = self.DOG(x4)
        x_4 = self.DOG(x3)
        x_8 = self.DOG(x2)

        return x_2, x_4, x_8, out

# if __name__ == "__main__":
#     test = torch.rand(4, 3, 480, 640)

#     net = ResDepth(max_depth=10, encoder_pretrained=True)

#     net(test)