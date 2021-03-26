import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class LinearAttentionBlock3D(nn.Module):
    def __init__(self, in_features, t_depth, normalize_attn=True):
        super(LinearAttentionBlock3D, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv3d(in_channels=in_features, out_channels=1, kernel_size=(t_depth,1,1), padding=0, bias=False)
    def forward(self, l):

        N, C, T, W, H = l.size()
        c = self.op(l) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,1,-1), dim=3).view(N,1,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,1,-1).sum(dim=3) # batch_sizexC
        else:
            g = F.adaptive_avg_pool3d(g, (1,1,1)).view(N,C,1)
        return c.view(N,1,1,W,H), g


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride_spatial in [1, 2]
        assert stride_temporal in [1, 2]
        assert kernel in [1, 3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        conv_layer = nn.Conv3d
        norm_layer = nn.BatchNorm3d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            # in, out, kernel, stride, padding
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3_ATTN(nn.Module):
    def __init__(self, n_class=368, input_size=112, dropout=0.8, mode='large', width_mult=1.0):
        super(MobileNetV3_ATTN, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting_1 = [
                # k, exp, c,  se,     nl,  s
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
            ]
            mobile_setting_2 = [
                # k, exp, c,  se,     nl,  s
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],

            ]

            mobile_setting_3 = [
                # k, exp, c,  se,     nl,  s
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        #assert input_size % 32 == 0
        #TODO: if large then do this else do that
        last_fc = 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features_1 = [conv_bn(3, input_channel, (1,2,2), nlin_layer=Hswish)]
        self.features_2 = []
        self.features_3 = []
        # self.features_4 = []
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s, st in mobile_setting_1:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features_1.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        for k, exp, c, se, nl, s, st in mobile_setting_2:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features_2.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        for k, exp, c, se, nl, s, st in mobile_setting_3:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features_3.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel


        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features_3.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features_4.append(nn.AdaptiveAvgPool3d(1))
            # self.features_4.append(nn.Conv3d(last_conv, last_channel, 1, 1, 0))
            # self.features_4.append(Hswish(inplace=True))
            last_fc = 40 + 112 + 960
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool3d(1))
            self.features.append(nn.Conv3d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features_1 = nn.Sequential(*self.features_1)
        self.features_2 = nn.Sequential(*self.features_2)
        self.features_3 = nn.Sequential(*self.features_3)
        # self.features_4 = nn.Sequential(*self.features_4)


        self.attn1 = LinearAttentionBlock(in_features=40, t_depth=4, normalize_attn=True)
        self.attn2 = LinearAttentionBlock(in_features=112, t_depth=2, normalize_attn=True)
        self.attn3 = LinearAttentionBlock(in_features=960, t_depth=1, normalize_attn=True)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_fc, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x_1 = self.features_1(x)
        x_2 = self.features_2(x_1)
        x_3 = self.features_3(x_2)
        # x_4 = self.features_4(x_3)

        ##### Attn Stuff #####
        c_1, ag_1 = self.attn1(x_1)
        c_2, ag_2 = self.attn2(x_2)
        c_3, ag_3 = self.attn3(x_3)

        g = torch.cat((ag_1,ag_2,ag_3), dim=1) # batch_sizexC
        ##### Attn Stuff #####

        x_4 = self.classifier(g.squeeze(-1))
        return x_4

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(**kwargs):
    model = MobileNetV3(**kwargs)
    return model

