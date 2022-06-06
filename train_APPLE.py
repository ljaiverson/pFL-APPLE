import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import create_parser, create_dir, set_seed
from utils.data_loader import get_federated_datasets_dict
from utils.models import Net, ClientNet
from utils.losses import reg_loss, get_reg_coef
from utils.apple import init_pss, prepare_client_model

import syft as sy 
hook = sy.TorchHook(torch)

import numpy as np

from collections import defaultdict
import copy
import os
import json

def train_one_round(args,
                    server_models,
                    client_models,
                    pss,
                    num_samples,
                    downloaded_once,
                    r,
                    criterion,
                    device,
                    central_server,
                    clients_pack_train):
    """
        This is a sequential simulation of training APPLE.
    """
    updated_server_models = []
    n_clients = len(clients_pack_train)
    p0 = torch.Tensor(init_pss(num_samples)[0]).to(device)
    p0.requires_grad = False
    for client_idx, (client, loader) in enumerate(clients_pack_train):
        # prepare client model for this client
        client_model = client_models[client_idx]
        client_model = prepare_client_model(args, pss, downloaded_once, client_model, client_idx, server_models, central_server, r)
        optimizer = optim.SGD([
                {'params': client_model.conv1s.parameters()},
                {'params': client_model.conv2s.parameters()},
                {'params': client_model.fc1s.parameters()},
                {'params': client_model.fc2s.parameters()},
                {'params': client_model.ps, 'lr': args.lr_coef * (args.decay ** r)}
            ], lr=args.lr_net * (args.decay ** r), momentum=0.9)

        # local training for one client
        for epoch in range(args.num_local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target.long()) + reg_loss(client_model.ps, p0, coef=get_reg_coef(args, r))
                loss.backward()
                optimizer.step()
                if (batch_idx+1) % args.log_step == 0 or (batch_idx + 1 == len(loader)):
                    print("******* Round: [%2d/%2d] | Client: [%3d/%3d] | Epoch [%3d/%3d] | Batch [%4d/%4d] | loss = %.8f *******" % 
                        (r+1, args.num_rounds,  client_idx+1, len(clients_pack_train), 
                            epoch+1, args.num_local_epochs, batch_idx+1, len(loader), loss.item()))

        client_model, ps = client_model.extract_learnables(client_idx)
        updated_server_models.append(client_model)
        pss[client_idx] = ps
        
    del server_models
    for m in updated_server_models:
        m = m.to(device)
        m.send(central_server)
    return updated_server_models, pss, client_models

def val_with_local_model(client_models, criterion, device, client_data_pack, mode="Test"):
    """
        This function computes the loss and acc for each client,
        as well as for the entire dataset (sum up all clients).
    """
    total_loss, total_correct, total_samples = 0, 0, 0
    clients_accs, clients_losses = [], []
    n_clients = len(client_models)
    for client_idx in range(n_clients):
        client_model = client_models[client_idx]
        client, loader = client_data_pack[client_idx]
        n_samples = len(loader.dataset)
        total_samples += n_samples
        client_correct = 0
        client_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                output = client_model(data)
                loss = (criterion(output, target.long()) * data.size(0)).item() # sum up batch loss
                client_loss += loss
                total_loss += loss
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
                correct = pred.eq(target.view_as(pred)).sum().item()
                client_correct += correct
                total_correct += correct

        client_loss /= n_samples
        clients_losses.append(client_loss)
        client_acc = client_correct / n_samples
        clients_accs.append(client_acc)
        print("==> {:5s} set: Client [{:2d}/{:2d}], Average loss: {:.4f}, Accuracy: {:6d}/{:6d} ({:.0f}%)".format(
            mode, client_idx+1, n_clients, client_loss, client_correct, n_samples, 100. * client_acc))
    total_loss /= total_samples
    acc = total_correct / total_samples
    print("==> {:5s} set: Average loss {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        mode, total_loss, total_correct, total_samples, 100. * acc))
    return total_loss, acc, clients_losses, clients_accs

def main():
    args = create_parser()
    in_channels = 1 if args.data in ["mnist", "organmnist_axial"] else 3
    if (args.data == "mnist" or args.data == "cifar10"):
        num_classes = 10
    elif args.data == "pathmnist":
        num_classes = 9
    elif args.data == "organmnist_axial":
        num_classes = 11
    create_dir(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datasets: {'client1': dataset1, ...}
    trainsets, valsets, testsets = get_federated_datasets_dict(args, True)

    # prepare client_pack: [(syft_client1, dataloader1), ...], and weights (for initial values of p's)
    central_server = sy.VirtualWorker(hook, id="server")
    clients_pack_train = []
    clients_pack_val = []
    clients_pack_test = []
    num_samples = []
    for client_id, trainset in trainsets.items():
        if client_id.startswith("entire"):
            continue
        num_samples.append(len(trainset))
        client = sy.VirtualWorker(hook, id=client_id)
        train_dloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dloader = DataLoader(dataset=valsets[client_id], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_dloader = DataLoader(dataset=testsets[client_id], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        clients_pack_train.append((client, train_dloader))
        clients_pack_val.append((client, val_dloader))
        clients_pack_test.append((client, test_dloader))

    # prepare model and loss
    prototype_model = Net(in_channels=in_channels, num_classes=num_classes).to(device)
    print()
    print(prototype_model)
    print()
    criterion = nn.CrossEntropyLoss()

    # train and test
    hist = defaultdict(lambda: [])

    # prepare initial model for each client
    server_models = []
    client_models = []
    pss = init_pss(num_samples)
    n_client = len(clients_pack_train)
    for client_idx, (client, _) in enumerate(clients_pack_train):
        # broadcast the model
        m = Net(in_channels=in_channels, num_classes=num_classes).to(device).send(central_server)
        server_models.append(m)
        client_model = ClientNet(n_client, in_channels, num_classes, pss[client_idx]).to(device)
        client_models.append(client_model)

    # train and test
    downloaded_once = (np.eye(n_client) == 1)
    for r in range(args.num_rounds):
        # train
        server_models, pss, client_models = train_one_round(args,
                                                            server_models,
                                                            client_models,
                                                            pss,
                                                            num_samples,
                                                            downloaded_once,
                                                            r,
                                                            criterion,
                                                            device,
                                                            central_server,
                                                            clients_pack_train)
        
        # val
        print()
        train_loss, train_acc, train_clients_losses, train_clients_accs = val_with_local_model(client_models,
                                                                                               criterion,
                                                                                               device,
                                                                                               clients_pack_train,
                                                                                               mode="Train")
        val_loss, val_acc, val_clients_losses, val_clients_accs = val_with_local_model(client_models,
                                                                                       criterion,
                                                                                       device,
                                                                                       clients_pack_val,
                                                                                       mode="Val")
        test_loss, test_acc, test_clients_losses, test_clients_accs = val_with_local_model(client_models,
                                                                                           criterion,
                                                                                           device,
                                                                                           clients_pack_test,
                                                                                           mode="Test")
        print()

        # store the result history
        if r == 0:
            hist["pss"].append(init_pss(num_samples))
        else:
            hist["pss"].append(copy.deepcopy(pss))
        hist['train_losses'].append(train_loss)
        hist['train_accs'].append(train_acc)
        hist['val_losses'].append(val_loss)
        hist['val_accs'].append(val_acc)
        hist["val_clients_accs"].append(val_clients_accs)
        hist["val_clients_mean_accs"].append(np.mean(val_clients_accs))
        hist["val_clients_losses"].append(val_clients_losses)
        hist['test_losses'].append(test_loss)
        hist['test_accs'].append(test_acc)
        hist["test_clients_accs"].append(test_clients_accs)
        hist["test_clients_mean_accs"].append(np.mean(test_clients_accs))
        hist["test_clients_losses"].append(test_clients_losses)

    # save the result history
    hist_result_fn = os.path.join(args.hist_dir, "%s-%s-APPLE.json" % (args.data, args.distribution))
    with open(hist_result_fn, 'w') as f:
        json.dump(hist, f)

    print("\n==> results saved at %s." % hist_result_fn)

if __name__ == '__main__':
    main()