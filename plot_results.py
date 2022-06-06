import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import json
import math
import os

def read_results(dsents_names, methods_names, distribution, path):
    res = {}
    for dn in dsents_names:
        if dn not in res:
            res[dn] = {}
        for mn in methods_names:
            with open(os.path.join(path, "%s-%s-%s.json" % (dn.lower(), distribution, mn)), "r") as f:
                res[dn][mn] = json.load(f)
    return res

def _compute_bmcta(data):
    data = np.array(data)
    m = np.mean(data, axis=1)
    return np.max(m), np.argmax(m)

def results_complete_summary(res_dict,
                             methods_names,
                             dsets_names,
                             fig_size=None,
                             bars_width=0.3,
                             distribution="non-iid-pathological",
                             show=True,
                             save=False,
                             fn=None,
                             dpi=600):
    """
        This function creates a 2D subplots with n_rows = 3 (in order: test accuarcy, training loss, client acc bar chart),
        and n_columns = n_datasets. It also prints out 2 2D matrices (n_datasets x n_methods), one for BMCTA, one for BTA
        and if save is True, then save the created plot at @argument fn.
        @argument res_dict: dict of (e.g. FedAvg, APPLE) {"mnist": {"FedAvg": hist_fedavg, "APPLE": hist_apple}} 
        @argument methods_names: methods_names representing the methods (e.g. FedAvg, APPLE)
        @argument bins_dict: provide bins as a dictionary, key is the name of the dataset that needs specific 
                             bins for the fairness plot.
        @return: None (but will )
    """
    res = []
    for dn in res_dict:
        dset_res = []
        for mn in methods_names:
            dset_res.append(res_dict[dn][mn])
        res.append(dset_res)

    client_labels = ["client {:d}".format(i+1) for i in range(12)]
    label_labelpad = 1.5
    n_dsets = len(res)
    n_methods = len(res[0])
    clients_pos = np.arange(len(client_labels)).astype(float)  # the label locations
    clients_pos *= 1.5 ** (math.ceil(n_methods / 2) - 1)
    if n_methods % 2 == 0:
        half_bar_poses = [i+0.5 for i in range(n_methods // 2)]
        bar_poses = [-i for i in half_bar_poses[::-1]] + half_bar_poses[:]
    else:
        half_bar_poses = [i+1 for i in range(n_methods // 2)]
        bar_poses = [-i for i in half_bar_poses[::-1]] + [0] + half_bar_poses[:]
    assert len(dsets_names) == n_dsets, "dsets_names does not equal to the number of datasets provided in argument res"
    BMTA_mat = []
    bestTestAcc_mat = []

    fig, ax = plt.subplots(3, n_dsets)
    fig_size = fig_size if fig_size is not None else (15, 8)
    fig.set_size_inches(fig_size)

    for i in range(n_dsets):
        BMTA_mat.append([])
        bestTestAcc_mat.append([])
        test_clients_accs = []
        clt_acc_y_min = 0.7
        clt_acc_y_max = 1.1
        ax[0][i].set_xlabel("Communication rounds", labelpad=label_labelpad)
        ax[0][i].grid()
        ax[0][i].set_title(dsets_names[i])
        ax[1][i].set_xlabel("Communication rounds", labelpad=label_labelpad)
        ax[1][i].grid()
        
        ax[2][i].set_xticks(clients_pos)
        ax[2][i].set_xticklabels(client_labels, rotation=60)
        for j in range(n_methods):
            bestTestAcc_mat[i].append(np.max(res[i][j]["test_accs"]))
            # test accuracy plot
            ax[0][i].plot(range(1, len(res[i][j]["test_accs"]) + 1), res[i][j]["test_accs"], label=methods_names[j])
            
            # training loss plot
            ax[1][i].plot(range(1, len(res[i][j]["train_losses"]) + 1), res[i][j]["train_losses"], label=methods_names[j])

            # client acc bar chart
            bmta, idx = _compute_bmcta(res[i][j]["test_clients_accs"])
            BMTA_mat[i].append(bmta)
            test_clients_accs.append(res[i][j]["test_clients_accs"][idx])
            pos = clients_pos + bars_width * bar_poses[j]
            rects = ax[2][i].bar(pos, test_clients_accs[j], bars_width, label=methods_names[j])
            
            if np.min(test_clients_accs[j]) - 0.1 < clt_acc_y_min:
                clt_acc_y_min = np.min(test_clients_accs[j]) - 0.1
        ax[2][i].set_ylim(clt_acc_y_min, clt_acc_y_max)


    # set plot limit
    if distribution == "non-iid-pathological":
        ax[0][0].set_ylim(bottom=0.92, top=1.005)
        ax[0][1].set_ylim(bottom=0.65)
        ax[0][2].set_ylim(bottom=0.82, top=1.02)
        ax[0][3].set_ylim(bottom=0.8, top=0.99)
        ax[1][0].set_ylim(bottom=-0.02, top=0.4)
        ax[1][1].set_ylim(top=1.0)
        ax[1][2].set_ylim(top=0.5)
        ax[1][3].set_ylim(bottom=-0.06, top=1.0)
    if distribution == "non-iid-practical":
        ax[0][0].set_ylim(bottom=0.85, top=1.007)
        ax[0][1].set_ylim(bottom=0.65, top=0.9)
        ax[0][3].set_ylim(bottom=0.7, top=0.96)
        ax[1][0].set_ylim(bottom=-0.07, top=0.9)
        ax[1][1].set_ylim(bottom=-0.1, top=1.5)
        ax[1][3].set_ylim(bottom=-0.07, top=2)
    
    # label padding
    ax[0][0].set_ylabel("Test accuracy", labelpad=label_labelpad)
    ax[1][0].set_ylabel("Training loss", labelpad=label_labelpad)
    ax[2][0].set_ylabel("Test accuracy", labelpad=label_labelpad)

    # labels
    handles, labels = ax[2][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=n_methods)

    # print BMCTA (best mean client-wise test accuracy) and BTA (best test accuracy)
    methods_str = "\t".join(methods_names)
    print("==> BMCTA for datasets\t%s" % methods_str)
    for i in range(n_dsets):
        print("%20s |\t" % dsets_names[i], "\t".join([("%2.2f" % (res*100)) for res in BMTA_mat[i]]))
    print("\n==> BTA for datasets\t%s" % methods_str)
    for i in range(n_dsets):
        print("%20s |\t" % dsets_names[i], "\t".join([("%2.2f" % (res*100)) for res in bestTestAcc_mat[i]]))
    print()

    # show and save
    if show:
        plt.show()
    if save:
        fig.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        print("==> figure saved at", fn)
    return

def plot_apple_ps_2x4(res, dsets_names, save=False, show=True, fn=None, dpi=600):
    """
        This function visualizes the trajectory of the p_i's. It creates a 2 x n_dataset(4) plot with
        the first row as the plots of the p_1 and the second row as the p_{i,i}'s.
    """
    n_dsets = len(dsets_names)
    fig, ax = plt.subplots(2, n_dsets)
    markers = [".", "^", "v", "P"]
    label_labelpad = 1.5
    fig.set_size_inches((17,9))
    sns.reset_orig()  # get default matplotlib styles back
    # cm = plt.get_cmap('gist_rainbow')
    method_name = "APPLE"
    for i in range(n_dsets):
        pss_hist = np.array(res[dsets_names[i]][method_name]["pss"])
        n_clients = len(pss_hist[0])
        clrs = sns.color_palette('husl', n_colors=n_clients)  # a list of RGB tuples
        ax[0][i].set_xlabel("Communication rounds", labelpad=label_labelpad)
        for j in range(n_clients):
            lines = ax[0][i].plot(range(1, len(pss_hist) + 1), np.abs(np.array(pss_hist[:, 0, j])),
                label=r'$p_{1,%d}$' % (j+1), marker=markers[j%len(markers)], markevery=10, markersize=4)
            lines[0].set_color(clrs[j])
        ax[0][i].grid()
        ax[0][i].legend(loc="right")
        ax[0][i].set_title(dsets_names[i])

        ax[1][i].set_xlabel("Communication rounds", labelpad=label_labelpad)
        ax[1][i].grid()
        for j in range(n_clients):
            lines = ax[1][i].plot(range(1, len(pss_hist) + 1), np.abs(np.array(pss_hist[:, j, j])), label=r'$p_{%d,%d}$' % (j+1,j+1))
            lines[0].set_color(clrs[j])
        ax[1][i].legend(loc="right")

    if show:
        plt.show()
    if save:
        fig.savefig(fn, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        print("==> figure saved at", fn)


def main():
    ##############################################################
    #  Modify the following block to adjust the results' output  #
    ##############################################################
    show = False # w.r.t. figures
    save = True # w.r.t. figures
    dpi = 600 # w.r.t. figures
    bars_width = 0.28
    distribution = "non-iid-pathological"
    results_folder = "./results/"
    results_folder = os.path.join(results_folder, distribution)
    dsets_names = [
        "MNIST",
        "CIFAR10",
        "OrganMNIST_axial",
        "PathMNIST",
    ]
    # methods_names = ["FedAvg", "FedAvg-localized", "APPLE"] # as an example
    methods_names = ["APPLE"]

    res = read_results(dsets_names, methods_names, distribution, results_folder)
    results_plots_dir = os.path.join(results_folder, "plots")
    if not os.path.exists(results_plots_dir):
        os.makedirs(results_plots_dir)
    print("**** figures (if being saved) would be saved at %s ****\n" % results_plots_dir)


    ################################### complete summary per distribution ####################################
    print("\n\n\n************ distribution:", distribution, "************")
    fn = "%s/complete-%s.png" % (results_plots_dir, distribution)
    results_complete_summary(res,
                            methods_names,
                            dsets_names,
                            fig_size=(16, 10),
                            bars_width=bars_width,
                            distribution=distribution,
                            show=show,
                            save=save,
                            fn=fn,
                            dpi=dpi)

    # # for plotting the DR vectors (var methods_names should only contain "APPLE")
    # fn = "%s/ps-for-apple-%s.png" % (results_plots_dir, distribution)
    # plot_apple_ps_2x4(res, dsets_names, save=save, show=show, fn=fn, dpi=dpi)


if __name__ == "__main__":
    main()
    