import torch.nn as nn

class C3D_Architecture(nn.Module):

    def __init__(self, num_classes=400) -> None:

        """This is a R(2+1)D Constructor Function
        Args:
            num_classes (int): Represents the number of output
            classes based on dataset
        Returns:
            None
        Raises:
            None
        """
        super(C3D_Architecture,self).__init__()

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
        #self.pool5 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = (0, 1, 1), ceil_mode = True)
        self.pool5 = nn.AdaptiveMaxPool3d((1,4,4))

        self.fc6 = nn.Linear(in_features = 8192, out_features = 4096)
        #self.bn6 = nn.BatchNorm1d(4096)
        self.fc7 = nn.Linear(in_features = 4096, out_features = 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.fc8 = nn.Linear(in_features = 4096, out_features = self.num_classes)

        # non-linear activations
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=0)

        # dropout
        self.dropout = nn.Dropout(p = 0.1)

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
        h = self.conv3b(h)
        h = self.relu(h)
        h = self.pool3(h)

        #4
        h = self.conv4a(h)
        h = self.relu(h)
        h = self.conv4b(h)
        h = self.relu(h)
        h = self.pool4(h)

        #5
        h = self.conv5a(h)
        h = self.relu(h)
        h = self.conv5b(h)
        h = self.relu(h)
        h = self.pool5(h)

        #linear layers

        h = h.view(-1, 8192)

        #6
        h = self.fc6(h)
        h = self.relu(h)
        #h = self.bn6(h)
        h = self.dropout(h)

        #7
        h = self.fc7(h)
        h = self.relu(h)
        h = self.bn7(h)
        h = self.dropout(h)

        #8
        h = self.fc8(h)
        #h = self.softmax(h)

        return h

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
            net_layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b, self.conv5a, self.conv5b, self.fc6, self.fc7, self.fc8]
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
            net_layers = [self.conv1, self.conv2, self.conv3a, self.conv3b, self.conv4a, self.conv4b, self.conv5a, self.conv5b, self.fc6, self.fc7, self.fc8]
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
