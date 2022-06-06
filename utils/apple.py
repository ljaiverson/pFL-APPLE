import torch
import numpy as np
import copy

def init_pss(num_samples):
    """
        This function initializes pss. pss[i] means the ps for client i.
        For all i, initial pss[i] equals a vector of normalized num_samples.
    """
    normed = [num_samples[i] / sum(num_samples) for i in range(len(num_samples))]
    pss = [normed[:] for i in range(len(num_samples))]
    return pss

def convert_weights_name(name, idx):
    l = name.split(".")
    return ".".join([l[0]+"s", str(idx), l[1]])

def prepare_client_model(args, pss, downloaded_once, client_model, client_idx, server_models, central_server, r):
    """
        This function assigns core models (selected core models, if bandwidth
        is limited) from server to client, while setting requires_grad to True
        only for client_idx portion (client's own core model).
    """
    n_clients = len(server_models)
    if (r == 0) or (args.limit_downloaded_models + 1 >= n_clients):
        # Here, programmed as "when current round==0, download all core models".
        # This is equivalent to assigning the same set of initial core models on every client.
        # This can be done by setting seeds in practice.
        download_which = list(range(n_clients))
    else:
        download_which = get_download_which(args, pss, downloaded_once, client_idx, r)
    with torch.no_grad():
        d_client = dict(client_model.named_parameters())
        d_client["ps"].requires_grad = True
        for i, m in enumerate(server_models):
            if not (i in download_which):
                continue
            m.get()
            for name, param in m.named_parameters():
                client_weights_name = convert_weights_name(name, i)
                if client_weights_name in d_client:
                    d_client[client_weights_name].data.copy_(param.data) # copy weights
                    d_client[client_weights_name].requires_grad = True if i == client_idx else False
            m.send(central_server)
    return client_model

def get_download_which(args, pss, downloaded_once, client_idx, r):
    """
        This function selects which M core models to download, under limited bandwidth.
    """
    download_which = [client_idx] # this shouldn't be considered as downloaded since already on client
    not_touched = np.where(downloaded_once[client_idx] == False)[0]
    ps_abs = np.abs(copy.deepcopy(pss[client_idx]))

    def choose_accord_ps(args, ps_abs, client_idx, choose_how_many, r):
        n_clients = len(ps_abs)
        mean_downloaded_times_per_model = r * args.limit_downloaded_models / n_clients
        base = max(args.thresh, mean_downloaded_times_per_model * args.lamb)
        mask = np.ones(n_clients, dtype=bool)
        mask[client_idx] = False
        ps_abs = ps_abs[mask]
        probs = base ** ps_abs
        probs = probs / np.sum(probs)
        l = np.arange(n_clients)[mask]
        d = np.random.choice(l, size=choose_how_many, replace=False, p=probs)
        return d.tolist()

    if len(not_touched) == 0:
        download_which += choose_accord_ps(args, ps_abs, client_idx, args.limit_downloaded_models, r)
    elif len(not_touched) >= args.limit_downloaded_models:
        np.random.shuffle(not_touched)
        download_which += not_touched.tolist()[:args.limit_downloaded_models]
    else:
        download_which += not_touched.tolist()
        download_which += choose_accord_ps(args, ps_abs, client_idx, args.limit_downloaded_models-len(not_touched), r)
    return download_which