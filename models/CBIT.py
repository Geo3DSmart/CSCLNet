from functools import partial
import timm

from models.help_funcs import *
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.UNetFormer import Block

class SUNet18(nn.Module):
    def __init__(self, in_ch, out_ch,
                 nonlinearity=partial(F.relu, inplace=True),
                 resnet=None,
                 share_encoder=False,
                 # resnet2 = models.resnet18(pretrained = True),
                 last_layer='tanh',
                 ):
        super().__init__()

        self.name = 'SUNet'

        resnet = timm.create_model('resnet18', pretrained=True, in_chans=in_ch)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.act1
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.conv_pred1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')


        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=out_ch)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.output_sigmoid = False
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, t1, t2):
        x11 = self.firstconv(t1)
        x11 = self.firstbn(x11)
        x11 = self.firstrelu(x11)
        xp11 = self.firstmaxpool(x11)
        xp12 = self.encoder1(xp11)
        skip12 = xp12
        xp13 = self.encoder2(xp12)
        skip13 = xp13
        xp14 = self.encoder3(xp13)
        skip14 = xp14
        xp15 = self.encoder4(xp14)
        skip15 = xp15

        x21 = self.firstconv(t2)
        x21 = self.firstbn(x21)
        x21 = self.firstrelu(x21)
        xp21 = self.firstmaxpool(x21)
        xp22 = self.encoder1(xp21)
        skip22 = xp22
        xp23 = self.encoder2(xp22)
        skip23 = xp23
        xp24 = self.encoder3(xp23)
        skip24 = xp24
        xp25 = self.encoder4(xp24)
        skip25 = xp25

        xd = self.conv5d(xp15, xp25)
        xd = self.conv4d(torch.cat((skip14, skip24), dim=1), xd)
        xd = self.conv3d(torch.cat((skip13, skip23), dim=1), xd)
        xd = self.conv2d(torch.cat((skip12, skip22), dim=1), xd)
        xd = self.conv1d(torch.cat((x11, x21), dim=1), xd)

        out2d = self.finaldeconv11(xd)
        out2d = self.finalrelu11(out2d)
        out2d = self.finalconv12(out2d)
        out2d = self.finalrelu12(out2d)
        out2d = self.finalconv13(out2d)
        out2d = self.finalnonlin1(out2d)

        out3d = self.finaldeconv21(xd)
        out3d = self.finalrelu21(out3d)
        out3d = self.finalconv22(out3d)
        out3d = self.finalrelu22(out3d)
        out3d = self.finalconv23(out3d)
        out3d = self.finalnonlin2(out3d)

        return out2d, out3d

    def forward_single(self, x):
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)

        xp11 = self.firstmaxpool(x1)
        xp12 = self.encoder1(xp11)
        xp13 = self.encoder2(xp12)
        xp14 = self.encoder3(xp13)

        xp11 = self.conv_pred1(xp11)

        return xp14, xp13, xp12, xp11


class CBIT(SUNet18):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable=False,
                 decoder_softmax=True,
                 ):

        super(CBIT, self).__init__(input_nc, output_nc)
        self.if_upsample_2x = if_upsample_2x

        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.learnable = learnable
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2 * dim

        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, self.token_len, 32))
        decoder_pos_size = [100, 50, 25]
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32, decoder_pos_size[0], decoder_pos_size[0]))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head

        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8, dim_head=self.dim_head, mlp_dim=mlp_dim, dropout=0)
        self.transformer_mac_encoder = TransformerDecoder(dim=dim, depth=self.dec_depth, heads=8,
                                                      dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,softmax=decoder_softmax)


        self.gltb = Block(dim=32, num_heads=4, window_size=4)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x):
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def _forward_mca_transformer_encoder(self, x, m):
        x = x + self.pos_embedding2
        x = self.transformer_mac_encoder(x, m)
        return x

    def bit(self,x1,x2):
        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
            token11 = self.transformer_mac_encoder(token2,token1)
            token22 = self.transformer_mac_encoder(token1,token2)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token11)
        x2 = self._forward_transformer_decoder(x2, token22)
        return x1,x2

    def forward(self, x1, x2):
        x1, x12, x13, x14 = self.forward_single(x1)
        x2, x22, x23, x24 = self.forward_single(x2)

        x1,x2 = self.bit(x1,x2)
        xd1 = x2 - x1

        x12,x22 = self.bit(x12,x22)
        xd2 = x22 - x12

        x13,x23 = self.bit(x13,x23)
        xd3 = x23 - x13

        x14,x24 = self.bit(x14,x24)
        xd4 = x24 - x14

        xd_sum = torch.cat([xd1, xd2], dim=1)
        xd_sum = self.conv_pred1(xd_sum)
        xd_sum = self.gltb(xd_sum)
        xd_sum = torch.cat([xd_sum, xd3], dim=1)
        xd_sum = self.conv_pred1(xd_sum)
        xd_sum = self.gltb(xd_sum)
        xd_sum = xd_sum + xd4

        r2d = xd_sum
        r3d = xd_sum

        if not self.if_upsample_2x:
            if self.learnable:
                r2d = self.upsamplex2l1(r2d)
                r2d = self.upsamplex2l2(r2d)
            else:
                r2d = self.upsamplex4(r2d)

        if self.learnable:
            r2d = self.upsamplex2l1(r2d)
            r2d = self.upsamplex2l2(r2d)
        else:
            r2d = self.upsamplex4(r2d)

        # forward small cnn
        x2d = self.classifier(r2d)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        if not self.if_upsample_2x:
            if self.learnable:
                r3d = self.upsamplex2l1(r3d)
                r3d = self.upsamplex2l2(r3d)
            else:
                r3d = self.upsamplex4(r3d)

        if self.learnable:
            r3d = self.upsamplex2l1(r3d)
            r3d = self.upsamplex2l2(r3d)
        else:
            r3d = self.upsamplex4(r3d)

        x3d = self.regressor(r3d)
        x3d = self.active3d(x3d)

        return x2d, x3d