import os
import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from .utils import find_files, find_subfolders, strip_suffix

scaled_bn_bools = {
    "scaled-False_bn-False": ("Unscaled", "no BN"),
    "scaled-False_bn-True": ("Unscaled", "BN"),
    "scaled-True_bn-False": ("Scaled (ours)", "no BN"),
    "scaled-True_bn-True": ("Scaled (ours)", "BN"),
}

tidy_dataset_name = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

tidy_arch_name = {"resnet32": "ResNet32", "resnet50": "ResNet50"}


def compute_ci_and_format(data, level=0):
    new_data = pd.DataFrame()
    for _, new_df in data.groupby(level=level):
        # new_df.columns = new_df.columns.droplevel(level=0)
        sorted_idx = np.argsort(new_df["mean"])
        is_gap = (
            new_df.iloc[sorted_idx[-1]]["lower_ci"]
            > new_df.iloc[sorted_idx[-2]]["upper_ci"]
        )
        bolds = [""] * len(sorted_idx)
        if is_gap:
            bolds[sorted_idx[-1]] = "\\bm"
        new_col = [
            "${}{{{:.2f}}}_{{\pm {:.2f}}}$".format(bold, mean, sigma)
            for bold, mean, sigma in zip(bolds, new_df["mean"], new_df["sigma"])
        ]
        series = pd.DataFrame(
            new_col,
            index=new_df.index,
            columns=["Test Accuracy"],
            dtype=pd.StringDtype(),
        )
        new_data = pd.concat([new_data, series])
    return new_data


def results_collect(dataset, arch, base_dir="runs"):
    summary_folder = os.path.join(base_dir, dataset, arch, "summary")
    results_folders = find_subfolders(summary_folder, prefix="scaled")
    # strip off learning rate, to select best learning rate with best accuracy
    experiment_folders = set(strip_suffix(results_folders, splt_str="_init-lr"))
    results = []
    for experiment_folder in experiment_folders:
        experiment_name = experiment_folder.rsplit("/", 1)[1]
        tune_folders = [
            folder for folder in results_folders if experiment_name in folder
        ]
        tuned_mean_acc, tuned_sigma = tune_best_acc(tune_folders)
        scaled, batch_norm = scaled_bn_bools[experiment_name]
        results.append(
            {
                "Dataset": tidy_dataset_name[dataset],
                "Scaled": scaled,
                "BatchNorm": batch_norm,
                "Architecture": tidy_arch_name[arch],
                "mean": tuned_mean_acc,
                "sigma": tuned_sigma,
                "lower_ci": tuned_mean_acc - tuned_sigma,
                "upper_ci": tuned_mean_acc + tuned_sigma,
            }
        )
    return pd.DataFrame(results)


def tune_best_acc(results_folders):
    # Return the best average accuracy across folders in results_folder
    best_mean = 0.0
    for results_folder in results_folders:
        curr_mean, curr_sigma = get_acc(results_folder)
        if curr_mean > best_mean:
            best_mean = curr_mean
            best_sigma = curr_sigma
    return best_mean, best_sigma


def get_acc(results_folder):
    # Return the average accuracy across runs in results_folder
    if not os.path.isdir(results_folder):
        raise ValueError(f"Results folder {results_folder} does not exist!")
    acc_files = find_files(results_folder, ending="best_acc.npy")
    accs = []
    for acc_file in acc_files:
        filepath = os.path.join(results_folder, acc_file)
        acc = np.load(filepath)
        acc = np.mean(acc)
        accs.append(acc)
    accs = np.array(accs)
    avg_acc = np.mean(accs)
    std = np.std(accs)
    sigma = 2 * std / np.sqrt(len(accs))
    return avg_acc, sigma


# def results_collect(init, scaled_init, bn, sp, act, depth):
#     training_mean = 0
#     training_std = 0
#     res_mean = 0
#     res_std = 0

#     results_path = init + "_sp" + str(sp).replace(".", "_")
#     if scaled_init:
#         results_path += "_scaled"
#     if bn:
#         results_path += "_bn"

#     results_path += "_" + act + "_" + str(depth)

#     print("this is the path")
#     print(results_path)
#     print(os.getcwd())

#     if not os.path.isdir(results_path):
#         print(results_path, "doesnt exist")
#         return [0, 0, [0], [0], [0]]

#     os.chdir(results_path)

#     # if not os.path.isfile('best_acc.npy'):
#     # 	print('dont have these results yet {}'.format(results_path))
#     # 	return [0, 0, [0], [0], [0]]

#     # res_vec = np.load('best_acc.npy')
#     # training_acc = np.load('test_acc.npy')
#     ratio_pruned = np.load("ratios_pruned.npy")

#     # training_mean = training_acc.mean(0)
#     # training_std = training_acc.std(0)

#     # print(res_vec.shape, 'this is the shape of the res_vec (should be 3)')

#     # res_mean = res_vec.mean()
#     # res_std = res_vec.std()

#     # print('We are using {}'.format(results_path))
#     # print(res_mean)
#     # print(res_std)

#     os.chdir("../")
#     return [res_mean, res_std, ratio_pruned, training_mean, training_std]


def plot_acc(tr_mean, tr_std):
    if _scaled_init:
        plt.plot(range(len(tr_mean)), tr_mean, label=init + "_scaled")
    else:
        plt.plot(range(len(tr_mean)), tr_mean, label=init)

    plt.fill_between(range(len(tr_mean)), tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)

    if init == "EOC" and scaled_init == True:
        [res_mean, res_std, ratio, tr_mean, tr_std] = results_collect(
            init, False, bn, sp, act, depth
        )
        plt.plot(range(len(tr_mean)), tr_mean, label=init)
        plt.fill_between(
            range(len(tr_mean)), tr_mean - tr_std, tr_mean + tr_std, alpha=0.2
        )


if __name__ == "__main__":
    dataset = "cifar10"
    arch = "resnet32"
    results_collect(dataset, arch, True, True, 1e-01)
    # _plot_acc = False
    # _plot_ratio = True
    # pics_dir = "/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/".format(arch)
    # if not os.path.isdir(pics_dir):
    #     os.mkdir(pics_dir)
    # if arch == "resnet32":
    #     exp_name = "cifar10_resnet32_SNIP"
    #     depth = 32
    # elif arch == "vgg19":
    #     # exp_name = 'cifar10_vgg19_SNIP_no_BN'
    #     exp_name = "cifar10_vgg19_SNIP_circular"
    #     depth = 19

    # dataset = "cifar10"
    # path_to_summary = "/data/ziz/ton/one_shot_pruning/GraSP/runs/pruning/{}/{}/{}/summary/".format(
    #     dataset, arch, exp_name
    # )
    # os.chdir(path_to_summary)

    # print(path_to_summary)

    # # Collect the results
    # # init = 'EOC'
    # # scaled_init = False
    # # bn = False
    # # sp = 0.999
    # #
    # # res_mean, res_std = results_collect(init, scaled_init, bn, sp)

    # init_vec = ["EOC", "ordered", "xavier"]  #'kaiming',
    # scaled_init = False
    # bn = False
    # # sp_vec = [0.0, 0.25, 0.50, 0.75, 0.98, 0.999]
    # sp_vec = [0.95, 0.995]
    # act_vec = ["tanh"]

    # results_matrix_mean = np.zeros([len(sp_vec), len(init_vec)])
    # results_matrix_std = np.zeros([len(sp_vec), len(init_vec)])

    # for act in act_vec:
    #     j = 0
    #     for sp in sp_vec:
    #         plt.figure()
    #         k = 0
    #         for init in init_vec:
    #             if init != "EOC":
    #                 _scaled_init = False
    #             else:
    #                 _scaled_init = scaled_init

    #             [res_mean, res_std, ratio, tr_mean, tr_std] = results_collect(
    #                 init, _scaled_init, bn, sp, act, depth
    #             )
    #             results_matrix_mean[j, k] = res_mean  # .round(2)
    #             results_matrix_std[j, k] = res_std  # .round(2)
    #             k += 1

    #             print("this is the ratio")
    #             print(ratio)
    #             if _plot_acc:
    #                 if len(tr_mean) == 1:
    #                     print("we continue")
    #                     continue
    #                 plot_acc(tr_mean, tr_std)
    #             elif _plot_ratio:
    #                 if _scaled_init:
    #                     plt.plot(ratio, label=init + "_scaled")
    #                 else:
    #                     plt.plot(ratio, label=init)

    #         if _plot_acc:
    #             # plt.ylim([25, 95])
    #             plt.legend()
    #             plt.title("ACC: We are using sp {} with act {}".format(sp, act))
    #             if not os.path.isdir(pics_dir + "/{}".format(act)):
    #                 os.mkdir(pics_dir + "/{}".format(act))
    #             plt.savefig(
    #                 "/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/{}/{}sp_act{}.png".format(
    #                     arch, act, sp, act
    #                 )
    #             )
    #         elif _plot_ratio:
    #             plt.legend()
    #             plt.title(
    #                 "Pruning Ratio: We are using sp {} with act {}".format(sp, act)
    #             )
    #             if not os.path.isdir(pics_dir + "/{}_ratio".format(act)):
    #                 os.mkdir(pics_dir + "/{}_ratio".format(act))
    #             plt.savefig(
    #                 "/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/{}_ratio/{}sp_act{}.png".format(
    #                     arch, act, sp, act
    #                 )
    #             )
    #         j += 1

    #     # scaled_init = False
    #     print("\n")
    #     print("We are using act {}".format(act))
    #     if scaled_init:
    #         print("We are using SCALED EOC")

    #     print("mean results {} x {}".format(init_vec, sp_vec))
    #     print(results_matrix_mean)
    #     print("\n")
    #     print("std results {} x {}".format(init_vec, sp_vec))
    #     print(results_matrix_std)

