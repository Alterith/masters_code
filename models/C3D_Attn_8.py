import torch
import torch.nn as nn
import torch.nn.functional as F

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

class C3D_ATTN_8_Architecture(nn.Module):

    def __init__(self, num_classes=368) -> None:

        """This is a R(2+1)D Constructor Function
        Args:
            num_classes (int): Represents the number of output
            classes based on dataset
        Returns:
            None
        Raises:
            None
        """
        super(C3D_ATTN_8_Architecture,self).__init__()

        #num_outputs
        self.num_classes = num_classes

        #layers
        self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 64, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = (0, 0, 0))

        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.pool2 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = (0, 0, 0))

        self.conv3a = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.conv3b = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.pool3 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = (0, 0, 0))

        self.conv4a = nn.Conv3d(in_channels = 256, out_channels = 512, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.conv4b = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.pool4 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = (0, 0, 0))

        self.conv5a = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        self.conv5b = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3, 3, 3), padding = (1, 1, 1), stride = 1)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1280, 900),
            nn.Linear(900, 600),
            nn.Linear(600, self.num_classes),
        )

        self.attn1 = LinearAttentionBlock3D(in_features=256, t_depth=4, normalize_attn=True)
        self.attn2 = LinearAttentionBlock3D(in_features=512, t_depth=2, normalize_attn=True)
        self.attn3 = LinearAttentionBlock3D(in_features=512, t_depth=1, normalize_attn=True)

        # non-linear activations
        self.relu = nn.ReLU()


    def forward(self, x):

        """This is a C3D forward pass function
        Args:
            x (torch tensor): The input video clip
        Returns:
            torch tensor: Representing the processed clip 'x'
        Raises:
            None
        """

        #1
        h = self.conv1(x)
        h = self.relu(h)
        h = self.pool1(h)

        #2
        h = self.conv2(h)
        h = self.relu(h)
        h = self.pool2(h)

        #3
        h = self.conv3a(h)
        h = self.relu(h)
        x_1 = self.conv3b(h)
        h = self.relu(x_1)
        h = self.pool3(h)

        #4
        h = self.conv4a(h)
        h = self.relu(h)
        x_2 = self.conv4b(h)
        h = self.relu(x_2)
        h = self.pool4(h)

        #5
        h = self.conv5a(h)
        h = self.relu(h)
        x_3 = self.conv5b(h)

        ##### Attn Stuff #####
        # spatial attn
        _, ag_1 = self.attn1(x_1)
        _, ag_2 = self.attn2(x_2)
        _, ag_3 = self.attn3(x_3)

        g = torch.cat((ag_1,ag_2,ag_3), dim=1) # batch_sizexC
        ##### Attn Stuff #####

        x_4 = self.classifier(g.squeeze(-1))
        return x_4

    def recursive_network_layer_extraction(self, net, layer_list):

            net_layers = (list(net.children()))
            for i, layer in enumerate(net_layers):
                # check if layer has children
                num_children = 0

                try:
                    num_children = len(list(layer.children()))
                except:
                    pass

                try:
                    if num_children  == 0 and "Conv" in str(layer) or "Linear" in str(layer):
                        layer_list.append(layer)
                    else:
                        layer_list = recursive_network_layer_extraction(layer, layer_list)
                except:
                    pass
                    #print(layer)
                    #print("Broken Function")

            return layer_list

    def recursive_network_layer_weight_load(self, net, layer_list, index, use_base_layers):
        if use_base_layers:
            net_layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b, self.conv5a, self.conv5b]
        else:
            net_layers = (list(net.children()))
        for i, layer in enumerate(net_layers):
                # check if layer has children
            num_children = 0

            try:
                num_children = len(list(layer.children()))
            except:
                pass

            try:
                if num_children  == 0 and "Conv" in str(layer) or "Linear" in str(layer):
                    layer.weight = layer_list[index].weight
                    index = index + 1
                elif (not "ReLU" in str(layer)):
                    index = recursive_network_layer_weight_load(layer, layer_list, index , 0)
                else:
                    pass
            except:
                print(layer)
                print("Broken Function")

        return index

    def recursive_network_layer_weight_check(self, net, layer_list, index, total, use_base_layers):
        if use_base_layers:
            net_layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b, self.conv5a, self.conv5b]
        else:
            net_layers = (list(net.children()))
        for i, layer in enumerate(net_layers):
                # check if layer has children
            num_children = 0

            try:
                num_children = len(list(layer.children()))
            except:
                pass

            try:
                if num_children  == 0 and "Conv" in str(layer) or "Linear" in str(layer):
                    #layer = layer_list[index]
                    layer_weight = torch.sum(layer.weight).item()
                    print(str(index),": ATTN_WEIGHT " , str(layer_weight))
                    layer_list_weight = torch.sum(layer_list[index].weight).item()
                    print(str(index),": EXISTING_WEIGHT " , str(layer_list_weight))
                    total += layer_weight - layer_list_weight
                    #print("broken_sum")
                    index = index + 1
                elif (not "ReLU" in str(layer)):
                    index, total = recursive_network_layer_weight_load(layer, layer_list, index, total, 0)
                else:
                    pass
            except:
                pass
                #print(layer)
                #print("Broken Function")

        return index, total
