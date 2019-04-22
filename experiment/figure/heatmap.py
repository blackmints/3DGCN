import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import Counter
from keras.models import Model
from rdkit import Chem
from rdkit.Chem import AllChem
from model.trainer import Trainer
from model.dataset import MPGenerator
from experiment.figure import Draw
from experiment.figure.Draw import DrawingOptions
from matplotlib import colors
import numpy as np
import csv, math


def draw_heatmap(path):
    # Choose closest trial to the average
    loss = []
    with open(path + '/raw_results.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "test_rmse" in row:
                loss.append(float(row["test_rmse"]))
            elif "test_roc" in row:
                loss.append(float(row["test_roc"]))
            else:
                raise ValueError("Cannot find average trial")

    avg = np.average(loss)
    idx = np.argmin(np.abs(np.array(loss) - avg))
    trial_path = path + "/trial_" + str(idx + 1)

    with open(trial_path + "/hyper.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper = dict(row)

    dataset = hyper['dataset']
    model = hyper['model']
    batch = int(hyper['batch'])
    units_conv = int(hyper['units_conv'])
    units_dense = int(hyper['units_dense'])
    num_layers = int(hyper['num_layers'])
    loss = hyper['loss']
    num_atoms = int(hyper['num_atoms'])
    pooling = hyper['pooling']
    std = float(hyper['data_std'])
    mean = float(hyper['data_mean'])

    # Load model
    trainer = Trainer(dataset)
    trainer.load_data(batch=batch)
    trainer.data.std = std
    trainer.data.mean = mean
    trainer.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers,
                       loss=loss, pooling=pooling)

    # Load best weight
    trainer.model.load_weights(trial_path + "/best_weight.hdf5")
    print("Loaded Weights from {}".format("/best_weight.hdf5"))

    # Load rotation test dataset
    trainer.data.replace_dataset(trial_path + "/test.sdf", subset="test", target_name="true")

    # Test set
    inputs_mol, inputs_cor = [], []
    for mol in Chem.SDMolSupplier(trial_path + "/test.sdf"):
        inputs_mol.append(mol)
        inputs_cor.append(mol.GetConformer().GetPositions())
    gen = MPGenerator(inputs_mol, inputs_cor, [1] * len(inputs_mol), 8, task="input_only", num_atoms=num_atoms)

    # Make submodel for retreiving features
    feature_model = Model(inputs=trainer.model.input, outputs=[trainer.model.get_layer("graph_conv_s_2").output,
                                                               trainer.model.get_layer("graph_conv_v_2").output])
    scalar_feature, vector_feature = feature_model.predict_generator(gen)
    print(scalar_feature.shape)

    # Parse feature to heatmap index
    scalar_feature = np.insert(scalar_feature, 0, 10e-6, axis=1)  # To find 0 column, push atom index by 1
    scalar_idx = np.argmax(scalar_feature, axis=1)

    vector_feature = np.sum(np.square(vector_feature), axis=2)
    vector_feature = np.insert(vector_feature, 0, 10e-6, axis=1)  # To find 0 column, push atom index by 1
    vector_idx = np.argmax(vector_feature, axis=1)

    scalar_idx_dict = []
    for scalar in scalar_idx:
        dic = Counter(scalar)
        if 0 in dic.keys():
            dic.pop(0)
        new_dic = {key - 1: value for key, value in dic.items()}

        idx = []
        for atom_idx in range(num_atoms):
            if atom_idx in new_dic:
                idx.append(new_dic[atom_idx] / units_conv)
            else:
                idx.append(0)
        scalar_idx_dict.append(idx)

    vector_idx_dict = []
    for vector in vector_idx:
        dic = Counter(vector)
        if 0 in dic.keys():
            dic.pop(0)
        new_dic = {key - 1: value for key, value in dic.items()}

        idx = []
        for atom_idx in range(num_atoms):
            if atom_idx in new_dic:
                idx.append(new_dic[atom_idx] / units_conv)
            else:
                idx.append(0)
        vector_idx_dict.append(idx)

    # Get 2D coordinates
    mols = []
    sdf = Chem.SDMolSupplier(trial_path + "/test.sdf")
    for mol in sdf:
        AllChem.Compute2DCoords(mol)
        mols.append(mol)

    DrawingOptions.bondLineWidth = 1.5
    DrawingOptions.elemDict = {}
    DrawingOptions.dotsPerAngstrom = 8
    DrawingOptions.atomLabelFontSize = 6
    DrawingOptions.atomLabelMinFontSize = 4
    DrawingOptions.dblBondOffset = 0.3
    cmap = colors.LinearSegmentedColormap.from_list("", ["white", "#fbcfb7", "#e68469", "#c03638"])

    for idx, (mol, scalar_dic, vector_dic) in enumerate(zip(mols[:20], scalar_idx_dict[:20], vector_idx_dict[:20])):
        fig = Draw.MolToMPL(mol, coordScale=1, size=(200, 200))
        x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=scalar_dic, step=0.0025)
        z = np.zeros((400, 400))
        z[:399, :399] = np.array(_z)[1:, 1:]
        max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))

        fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                           vmin=0, vmax=max_scale)
        fig.axes[0].set_axis_off()
        fig.savefig("./vis/{}/test_{}_scalar.png".format(dataset, idx), bbox_inches='tight')
        fig.clf()

        fig = Draw.MolToMPL(mol, coordScale=1, size=(200, 200))
        x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=vector_dic, step=0.0025)
        z = np.zeros((400, 400))
        z[:399, :399] = np.array(_z)[1:, 1:]
        max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))

        fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                           vmin=0, vmax=max_scale)
        fig.axes[0].set_axis_off()
        fig.savefig("./vis/{}/test_{}_vector.png".format(dataset, idx), bbox_inches='tight')
        fig.clf()

        fig = Draw.MolToMPL(mol, coordScale=1, size=(200, 200))
        x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=np.add(scalar_dic, vector_dic) / 2, step=0.0025)
        z = np.zeros((400, 400))
        z[:399, :399] = np.array(_z)[1:, 1:]
        max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))

        fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                           vmin=0, vmax=max_scale)
        fig.axes[0].set_axis_off()
        fig.savefig("./vis/{}/test_{}_merge.png".format(dataset, idx), bbox_inches='tight')
        fig.clf()
