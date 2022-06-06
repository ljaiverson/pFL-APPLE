import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
        This is the structure for the core models (wc),
        and the personalized models (wp) used for test time.
        Although wp = p1*wc1 + p2*wc2 + p3*wc3 + ..., but the
        core model weights and the coefficients (DRs) cannot
        be found in this structure, since this is the
        resulted wp (see method 'compressed' in class ClientNet).
    """
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClientNet(nn.Module):
    """
        This is the structure for the personalized models (wp)
        that used for training phase. It is a convex combination
        of the core models, i.e. wp = p1*wc1 + p2*wc2 + p3*wc3 + ...
        Core model weights and the coefficients can be found in this
        structure. Use method 'compressed' to compute the final wp
        in class Net.
    """
    def __init__(self, n_client, in_channels, num_classes, ps=None):
        super(ClientNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        ps = [1/n_client for _ in range(n_client)] if ps == None else ps
        
        self.conv1s = nn.ModuleList([nn.Conv2d(in_channels, 20, 5, 1) for i in range(n_client)])
        self.conv2s = nn.ModuleList([nn.Conv2d(20, 50, 5, 1) for i in range(n_client)])
        self.fc1s = nn.ModuleList([nn.Linear(4*4*50, 500) for i in range(n_client)])
        self.fc2s = nn.ModuleList([nn.Linear(500, num_classes) for i in range(n_client)])
        self.ps = nn.Parameter(torch.Tensor(ps))

    def superBlock(self, x, layers):
        y = 0
        for i, lay in enumerate(layers):
            y += lay(x) * self.ps[i]
        return y

    def forward(self, x):
        x = self.superBlock(x, self.conv1s)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.superBlock(x, self.conv2s)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.superBlock(x, self.fc1s)
        x = F.relu(x)
        x = self.superBlock(x, self.fc2s)
        return x
    
    def _convert_weights_name(self, name, idx):
        l = name.split(".")
        return ".".join([l[0]+"s", str(idx), l[1]])

    def compress(self):
        """
            This method converts a class ClientNet personalized model
            into a class Net personalized model, in other words, compute
            p1*wc1 + p2*wc2 + p3*wc3 + ... into wp.
        """
        ps = self.ps.data.clone().tolist()
        d_client_model = dict(self.named_parameters())
        with torch.no_grad():
            updated_model = Net(in_channels=self.in_channels, num_classes=self.num_classes)
            for name, params in updated_model.named_parameters():
                value = 0
                for client_idx in range(len(ps)):
                    value += ps[client_idx] * d_client_model[self._convert_weights_name(name, client_idx)].data
                params.data.copy_(value)
        return updated_model

    def extract_learnables(self, client_idx):
        """
            This function returns all local learnables, i.e. 
            the updated local core model (this client_idx's own core model),
            and the Directed Relationship vector ps.
        """
        ps = self.ps.data.clone().tolist()
        d_client_model = dict(self.named_parameters())
        with torch.no_grad():
            updated_model = Net(in_channels=self.in_channels, num_classes=self.num_classes)
            for name, params in updated_model.named_parameters():
                params.data.copy_(d_client_model[self._convert_weights_name(name, client_idx)].data)
        return updated_model, ps
