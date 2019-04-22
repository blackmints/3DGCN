import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from rdkit import Chem
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.size'] = 16
plt.rcParams['axes.axisbelow'] = True

red, orange, green, blue, weave = "#CC3311", "#ED7D0F", "#009988", "#0077BB", "#aaaaaa"
color = {"delaney": "#ED7D0F", "freesolv": "#CC3311", "bace_reg": "#0077BB"}
tick = {"delaney": 3.0, "freesolv": 3.0}


def walk_level(path, level=1):
    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)

    for root, dirs, files in os.walk(path):
        yield root, dirs, files

        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def find_best_hyper(dataset, model, metric="test_rmse"):
    path = "../../result/{}/{}/".format(model, dataset)

    # Get list of hyperparameters
    names, losses, stds = [], [], []
    for root, dirs, files in walk_level(path, level=0):
        for dir_name in dirs:
            loss = []
            if os.path.isfile(path + dir_name + "/results.csv"):
                with open(path + dir_name + "/results.csv") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        loss.append(row[metric])

                names.append(dir_name)
                losses.append(float(loss[0]))
                stds.append(float(loss[1]))

    # Sort by loss
    losses, stds, names = zip(*sorted(zip(losses, stds, names)))

    # Choose lowest loss hyper
    path += names[np.argmin(losses)] + '/'

    return path


def find_average_trial(dataset, model, metric="test_rmse"):
    path = find_best_hyper(dataset, model, metric=metric)

    # Choose closest trial to the average
    loss = []
    with open(path + '/raw_results.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss.append(float(row[metric]))

    avg = np.average(loss)
    idx = np.argmin(np.abs(np.array(loss) - avg))
    path += 'trial_' + str(idx + 1) + '/'

    return path


def draw_confusion_graph(dataset, base_path):
    c = color[dataset]
    t = tick[dataset]

    histo, histo_weave = [], []
    for i in range(1, 11):
        path = base_path + "trial_{}/".format(i)

        # Load true, pred value
        true_y, pred_y, diff_y, weave_y, weave_diff_y = [], [], [], [], []

        if os.path.isfile(path + "weave_trial_{}.sdf".format(i)):
            is_weave = True
            mols = Chem.SDMolSupplier(path + "weave_trial_{}.sdf".format(i))
            for mol in mols:
                if "true" not in mol.GetPropNames():
                    continue
                true_y.append(float(mol.GetProp("true")))
                pred_y.append(float(mol.GetProp("pred")))
                weave_y.append(float(mol.GetProp("pred_weave")))
                diff_y.append(float(mol.GetProp("true")) - float(mol.GetProp("pred")))
                weave_diff_y.append(float(mol.GetProp("true")) - float(mol.GetProp("pred_weave")))
        else:
            is_weave = False
            mols = Chem.SDMolSupplier(path + "test.sdf")
            for mol in mols:
                true_y.append(float(mol.GetProp("true")))
                pred_y.append(float(mol.GetProp("pred")))
                diff_y.append(float(mol.GetProp("true")) - float(mol.GetProp("pred")))

        true_y = np.array(true_y, dtype=float)
        pred_y = np.array(pred_y, dtype=float)
        diff_y = np.array(diff_y, dtype=float)
        weave_y = np.array(weave_y, dtype=float)
        weave_diff_y = np.array(weave_diff_y, dtype=float)
        histo += list(np.abs(diff_y))
        histo_weave += list(np.abs(weave_diff_y))

        # Generate linear trend line
        trend_z = np.polyfit(true_y, pred_y, 1)
        trend_p = np.poly1d(trend_z)

        if is_weave:
            trend_weave_z = np.polyfit(true_y, weave_y, 1)
            trend_weave_p = np.poly1d(trend_weave_z)

        # Find largest, smallest error molecules
        idx = np.argsort(diff_y)
        top_1 = mols[int(idx[-1])]
        top_2 = mols[int(idx[-2])]
        btm_1 = mols[int(idx[0])]
        btm_2 = mols[int(idx[1])]

        best_idx = np.argsort(np.abs(diff_y))
        best = mols[int(best_idx[0])]

        # Generate canvas
        plt.figure(figsize=(4.8, 4.8))

        # Plot points
        if is_weave: plt.scatter(true_y, weave_y, color=weave, s=np.pi * 10, alpha=0.8, linewidths=0)
        plt.scatter(true_y, pred_y, color=c, s=np.pi * 10, alpha=0.8, linewidths=0)
        x_min, x_max, y_min, y_max = plt.axis()

        # Make plotting box square
        plt.axis([min(x_min, y_min), max(x_max, y_max), min(x_min, y_min), max(x_max, y_max)])
        plt.xticks(np.arange(int(min(x_min, y_min)), max(x_max, y_max), t))
        plt.yticks(np.arange(int(min(x_min, y_min)), max(x_max, y_max), t))
        x_min, x_max, y_min, y_max = plt.axis()

        # Plot trend and identity line
        if is_weave: plt.plot([x_min, x_max], [trend_weave_p(x_min), trend_weave_p(x_max)], color=weave, alpha=0.8,
                              linestyle="-")
        plt.plot([x_min, x_max], [trend_p(x_min), trend_p(x_max)], color=c, alpha=0.8, linestyle="-")
        plt.plot([x_min, x_max], [x_min, x_max], color='black', alpha=0.5, linestyle="--")

        plt.ylabel('Predicted Free Energy')
        plt.xlabel('Experimental Free Energy')

        fig_name = path + "confusion_plot_" + dataset + "_trial" + str(i) + ".png"
        plt.savefig(fig_name, dpi=600)

        # Save example molecules
        writer = Chem.SDWriter(path + "confusion_examples_" + dataset + "_trial" + str(i) + ".sdf")
        for mol in [top_1, top_2, btm_1, btm_2, best]:
            writer.write(mol)

        print("Confusion Plot figure saved on {}".format(fig_name))

        plt.clf()

    def to_percent(y, position):
        # The percent symbol needs escaping in latex
        if plt.rcParams['text.usetex'] is True:
            return str(int(y * 100)) + r'$\%$'
        else:
            return str(int(y * 100)) + '%'

    # Draw error histogram
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    fig.subplots_adjust(hspace=0.3)

    if dataset == "freesolv":
        bins_x = np.arange(0, np.ceil(np.max(histo)), 0.5)
        bins_y = np.arange(0, 0.41, 0.2)
    elif dataset == "delaney":
        bins_x = np.arange(0, np.ceil(np.max(histo)), 0.5)
        bins_y = np.arange(0, 0.41, 0.2)

    weights = np.ones_like(histo) / float(len(histo))
    weaave_weights = np.ones_like(histo_weave) / float(len(histo_weave))
    ax[1].hist(histo_weave, color=weave, bins=np.arange(0, np.ceil(np.max(histo)), 0.25), weights=weaave_weights)
    ax[0].hist(histo, color=c, bins=np.arange(0, np.ceil(np.max(histo)), 0.25), weights=weights)
    ax[0].set_xticks(bins_x)
    ax[0].set_yticks(bins_y)
    ax[1].set_xticks(bins_x)
    ax[1].set_yticks(bins_y)

    formatter = FuncFormatter(to_percent)
    ax[0].yaxis.set_major_formatter(formatter)
    ax[1].yaxis.set_major_formatter(formatter)

    fig_name = base_path + "confusion_plot_histogram_" + dataset + ".png"
    plt.savefig(fig_name, dpi=600)

    print("Histogram Plot figure saved on {}".format(fig_name))


def find_confusion(dataset, base_path):
    for i in range(1, 11):
        path = base_path + "trial_{}/".format(i)

        # Load true, pred value
        true_y, pred_y, diff_y = [], [], []

        mols = Chem.SDMolSupplier(path + "test.sdf")
        for mol in mols:
            diff_y.append(float(mol.GetProp("true")) - float(mol.GetProp("pred")))

        diff_y = np.array(diff_y, dtype=float)

        # Find largest, smallest error molecules
        idx = np.argsort(diff_y)
        top_1 = mols[int(idx[-1])]
        top_2 = mols[int(idx[-2])]
        btm_1 = mols[int(idx[0])]
        btm_2 = mols[int(idx[1])]

        best_idx = np.argsort(np.abs(diff_y))
        best = mols[int(best_idx[0])]

        # Save example molecules
        writer = Chem.SDWriter(path + "confusion_examples_" + dataset + "_trial" + str(i) + ".sdf")
        for mol in [top_1, top_2, btm_1, btm_2, best]:
            writer.write(mol)

