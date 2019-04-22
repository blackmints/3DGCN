import matplotlib.pyplot as plt
import numpy as np
import os
from rdkit import Chem
from experiment.figure.confusion_plot import find_average_trial
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

plt.rcParams['font.size'] = 16
plt.rcParams['axes.axisbelow'] = True
red, orange, green, blue, weave = "#CC3311", "#ED7D0F", "#009988", "#0077BB", "#cccccc"


def draw_pr_curve(dataset, base_path):
    if dataset == "bace_cla":
        c = green
    else:
        c = blue
    for i in range(5, 6):
        path = base_path + "trial_{}/".format(i)
        # Load true, pred value
        true_y, pred_y, weave_y = [], [], []

        if os.path.isfile(path + "weave_trial_{}.sdf".format(i)):
            is_weave = True
            mols = Chem.SDMolSupplier(path + "weave_trial_{}.sdf".format(i))
            for mol in mols:
                if "true" not in mol.GetPropNames():
                    continue
                true_y.append(float(mol.GetProp("true")))
                pred_y.append(float(mol.GetProp("pred")))
                weave_y.append(1 - float(mol.GetProp("pred_weave")))
        else:
            is_weave = False
            mols = Chem.SDMolSupplier(path + "test.sdf")
            for mol in mols:
                true_y.append(float(mol.GetProp("true")))
                pred_y.append(float(mol.GetProp("pred")))

        true_y = np.array(true_y, dtype=float)
        pred_y = np.array(pred_y, dtype=float)
        weave_y = np.array(weave_y, dtype=float)

        # Get roc / precision and recall
        precision, recall, _ = precision_recall_curve(true_y, pred_y)
        fpr, tpr, _ = roc_curve(true_y, pred_y)

        if is_weave:
            precision_w, recall_w, _ = precision_recall_curve(true_y, weave_y)
            fpr_w, tpr_w, _ = roc_curve(true_y, weave_y)

        print("ROC 3DGCN: {}".format(roc_auc_score(true_y, pred_y)))
        if is_weave: print("ROC Weave: {}".format(roc_auc_score(true_y, weave_y)))
        print("PR 3DGCN: {}".format(average_precision_score(true_y, pred_y)))
        if is_weave: print("PR Weave: {}".format(average_precision_score(true_y, weave_y)))

        # Generate canvas
        w, h = plt.figaspect(1)
        plt.figure(figsize=(w, h))

        # Draw ROC curve
        if is_weave: plt.step(fpr_w, tpr_w, color=weave, alpha=1, where='post')
        plt.step(fpr, tpr, color=c, alpha=1, where='post')

        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

        fig_name = path + "ROC_curve_trial" + str(i) + ".png"
        plt.savefig(fig_name, dpi=600)
        plt.clf()
        print("ROC curve figure saved on {}".format(fig_name))

        # Draw PR curve
        plt.figure(figsize=(w, h))
        if is_weave: plt.step(recall_w, precision_w, color=weave, alpha=1, where='post')
        plt.step(recall, precision, color=c, alpha=1, where='post')

        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

        fig_name = path + "PR_curve_trial" + str(i) + ".png"
        plt.savefig(fig_name, dpi=600)
        plt.clf()
        print("PR curve figure saved on {}".format(fig_name))


def draw_confusion_matrix(dataset, model, set_trial=None, filename="test_results.sdf"):
    path = find_average_trial(dataset, model, metric="test_pr") if set_trial is None \
        else "../result/{}/{}/{}/".format(model, dataset, set_trial)

    # Load true, pred value
    true_y, pred_y = [], []
    mols = Chem.SDMolSupplier(path + filename)

    for mol in mols:
        true_y.append(float(mol.GetProp("true")))
        pred_y.append(float(mol.GetProp("pred")))

    true_y = np.array(true_y, dtype=float)
    pred_y = np.array(pred_y, dtype=float).round()

    # Get precision and recall
    confusion = confusion_matrix(true_y, pred_y)
    tn, fp, fn, tp = confusion.ravel()

    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
