from functools import partial
import timm

from .help_funcs import *

class SUNet18(nn.Module):
    def __init__(self, in_ch, out_ch,
        nonlinearity =  partial(F.relu, inplace=True),
        resnet = None,
        share_encoder = False,
        # resnet2 = models.resnet18(pretrained = True),
        last_layer = 'tanh',
        ):

        super().__init__()

        self.name = 'SUNet'
        # self.share_encoder = share_encoder

        C = [32, 64, 128, 256, 512, 1024]

        # resnet1 = resnet
        resnet = timm.create_model('resnet18', pretrained = True, in_chans = in_ch)

        # Encoder first (and second) image
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.act1
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        
        # Decoder
        # self.conv5d = DecBlock(C[4], C[4], C[3])
        # self.conv4d = DecBlock(C[3]+C[3], C[3], C[2])
        # self.conv3d = DecBlock(C[2]+C[2], C[2], C[1])
        # self.conv2d = DecBlock(C[1]+C[1], C[1], C[1])
        # self.conv1d = DecBlock(C[1]+C[1], C[1], C[0]) #, out_ch, bn=False, act=False)

        # self.convch = DecBlock(C[1], C[1], C[0])
        # self.convch2 = DecBlock(C[4], C[4], C[3])
        # self.convch3 = DecBlock(C[3], C[3], C[2])
        # self.convch4 = DecBlock(C[2], C[2], C[1])
        # self.convch5 = DecBlock(C[1], C[1], C[0])
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.depth_reduction = nn.Conv2d(256, 32, kernel_size=1)

        # self.finaldeconv11 = nn.ConvTranspose2d(C[0], 32, 4, 2, 1)
        # self.finalrelu11 = nonlinearity
        # self.finalconv12 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu12 = nonlinearity
        self.finalconv13 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.finalnonlin1 = nn.LogSoftmax(dim=1)

        # self.finaldeconv21 = nn.ConvTranspose2d(C[0], 32, 4, 2, 1)
        # self.finalrelu21 = nonlinearity
        # self.finalconv22 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu22 = nonlinearity
        # self.finalconv23 = nn.Conv2d(32, 1, 2, stride = 2, padding=0)
        self.finalconv23 = nn.Conv2d(32, 1, kernel_size= 3, stride = 1, padding=1) #kernel= 2cambiare stride in 2 e pad 0
        self.finalnonlin2 = last_activation(last_layer)



    def forward(self, t1, t2):
        #Encode branch 1
        # Stage 1
        x11 = self.firstconv(t1)
        x11 = self.firstbn(x11)
        x11 = self.firstrelu(x11)

        xp11 = self.firstmaxpool(x11) 

        # Stage 2
        xp12 = self.encoder1(xp11)
        skip12 = xp12

        # Stage 3
        xp13 = self.encoder2(xp12)
        skip13 = xp13

        # Stage 4
        xp14 = self.encoder3(xp13)
        skip14 = xp14

        # Stage 5
        # xp15 = self.encoder4(xp14)
        # skip15 = xp15

        # Encode branch 2

        # Stage 1
        x21 = self.firstconv(t2)
        x21 = self.firstbn(x21)
        x21 = self.firstrelu(x21)

        xp21 = self.firstmaxpool(x21) 

        #Stage 2
        xp22 = self.encoder1(xp21)
        skip22 = xp22

        # Stage 3
        xp23 = self.encoder2(xp22)
        skip23 = xp23

        # Stage 4
        xp24 = self.encoder3(xp23)
        skip24 = xp24

        xp15 = self.depth_reduction(xp14)
        xp25 = self.depth_reduction(xp24)
        xp15 = self.upsample(xp15)
        xp25 = self.upsample(xp25)
        xd = xp25 - xp15
        out2d = self.finalconv13(xd)
        out2d = self.finalnonlin1(out2d)

        out3d = self.finalconv23(xd)
        out3d = self.finalnonlin2(out3d)

        return out2d, out3d