import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import MultipleLocator, FixedFormatter, NullFormatter
from model.trainer import Trainer
from model.callback import calculate_roc_pr
from model.dataset import MPGenerator
from rdkit import Chem

plt.rcParams['font.size'] = 16
plt.rcParams['axes.axisbelow'] = True


def load_model(trial_path):
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
    print("Loaded Weights from {}".format(trial_path + "/best_weight.hdf5"))

    return trainer, hyper


def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    r_x = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_y = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_z = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    elif axis == "y":
        r = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    elif axis == "z":
        r = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r


def rotation_prediction(path, rotation="random", axis="", degree=""):
    # Iterate over trials
    raw_results = []
    task = None
    for i in range(1, 11):
        trial_path = path + "/trial_" + str(i)
        trainer, hyper = load_model(trial_path)

        # Rotate test dataset
        mols = Chem.SDMolSupplier(trial_path + "/test.sdf")

        rotated_mols = []
        print("Rotating Molecules... Rule: {}".format(rotation))
        for mol in mols:
            if rotation == "random":
                rotation_matrix = random_rotation_matrix()
            elif rotation == "stepwise":
                rotation_matrix = degree_rotation_matrix(axis, float(degree))
            else:
                raise ValueError("Unsupported rotation mechanism: {}".format(rotation))

            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()

                pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
                pos_rotated = np.matmul(rotation_matrix, pos)

                mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
            rotated_mols.append(mol)

        # Save rotated test dataset
        w = Chem.SDWriter(trial_path + "/test_" + rotation + axis + str(degree) + ".sdf")
        for m in rotated_mols:
            if m is not None:
                w.write(m)

        # Load rotation test dataset
        trainer.data.replace_dataset(trial_path + "/test_" + rotation + axis + str(degree) + ".sdf",
                                     subset="test", target_name="true")

        # Predict
        if hyper["loss"] == "mse":
            test_loss = trainer.model.evaluate_generator(trainer.data.generator("test"))
            pred = trainer.model.predict_generator(trainer.data.generator("test", task="input_only"))
            raw_results.append([test_loss[1], test_loss[2]])

        else:
            roc, pr, pred = calculate_roc_pr(trainer.model, trainer.data.generator("test"), return_pred=True)
            raw_results.append([roc, pr])

        # Save results
        trainer.data.save_dataset(trial_path + "/", pred, target="test",
                                  filename="test_" + rotation + axis + str(degree))

    # Save results
    results_mean = np.array(raw_results).mean(axis=0)
    results_std = np.array(raw_results).std(axis=0)

    header = ["test_mae", "test_rmse"] if task == "regression" else ["test_roc", "test_pr"]
    with open(path + "/rotation_" + rotation + axis + str(degree) + ".csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        writer.writerow(results_mean)
        writer.writerow(results_std)

    with open(path + "/rotation_raw_" + rotation + axis + str(degree) + ".csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        for r in raw_results:
            writer.writerow(r)


def single_molecule_rotation(path):
    # Iterate over trials
    for i in range(1, 11):
        trial_path = path + "/trial_" + str(i)
        trainer, hyper = load_model(trial_path)

        # Load test dataset
        trainer.data.replace_dataset(trial_path + "/test.sdf", subset="test", target_name="true")

        # Find example molecule
        active = []
        inactive = []
        mols = Chem.SDMolSupplier(trial_path + "/test.sdf")
        print("Finding example molecules...")
        for mol in mols:
            if "true" in mol.GetPropNames():
                if hyper["dataset"] == "delaney" or hyper["dataset"] == "freesolv":
                    active.append(mol)
                    inactive.append(mol)
                else:
                    if mol.GetProp("true") == "1" and float(mol.GetProp("pred")) >= 0.5:
                        active.append(mol)
                    elif mol.GetProp("true") == "0" and float(mol.GetProp("pred")) < 0.5:
                        inactive.append(mol)

        min_atoms1 = min([mol.GetNumAtoms() for mol in active])
        min_atoms2 = min([mol.GetNumAtoms() for mol in inactive])

        active_mols, inactive_mols = [], []
        for mol in active:
            if mol.GetNumAtoms() <= min_atoms1 + 5:
                active_mols.append(mol)

        for mol in inactive:
            if mol.GetNumAtoms() <= min_atoms2 + 15:  # bace_rotated require 6
                inactive_mols.append(mol)

        print("Found {} / {} molecules.".format(len(active_mols), len(inactive_mols)))

        w = Chem.SDWriter(trial_path + "/test_active.sdf")
        for mol in active_mols:
            w.write(mol)
        w = Chem.SDWriter(trial_path + "/test_inactive.sdf")
        for mol in inactive_mols:
            w.write(mol)
        print("Saved example molecules.")

        mols = active_mols[:5] + inactive_mols[:5]

        # Predict single molecule rotation
        for axis in ["x", "y", "z"]:
            print("Rotating along {}...".format(axis))
            example_results = []
            for active_mol in mols:
                inputs_mol, inputs_cor = [], []
                for degree in np.arange(0, 360, 5):
                    mol = rotate_molecule(active_mol, axis, degree)
                    inputs_mol.append(mol)
                    inputs_cor.append(mol.GetConformer().GetPositions())
                gen = MPGenerator(inputs_mol, inputs_cor, [1] * len(inputs_mol), 16, task="input_only",
                                  num_atoms=int(hyper["num_atoms"]))
                example_results.append(trainer.model.predict_generator(gen).flatten())

            with open(trial_path + "/rotation_single_{}.csv".format(axis), "w") as file:
                writer = csv.writer(file, delimiter=",")
                for row in example_results:
                    writer.writerow(row)


def bace_overlap_single_rotation():
    # Iterate over trials
    for i in range(1, 11):
        trainer, hyper = load_model("../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018/trial_4")
        trainer2, hyper2 = load_model(
            "../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113/trial_{}".format(i))

        # Load test dataset
        trainer.data.replace_dataset("../../experiment/figure/bace_overlap_{}.sdf".format(i), subset="test",
                                     target_name="true")
        trainer2.data.replace_dataset("../../experiment/figure/bace_overlap_{}.sdf".format(i), subset="test",
                                      target_name="true")

        # Find example molecule
        active = []
        inactive = []
        mols = Chem.SDMolSupplier("../../experiment/figure/bace_overlap_{}.sdf".format(i))
        print("Finding example molecules...")
        for mol in mols:
            if "true" in mol.GetPropNames():
                if mol.GetProp("true") == "1" and float(mol.GetProp("pred")) >= 0.5:
                    active.append(mol)
                elif mol.GetProp("true") == "0" and float(mol.GetProp("pred")) < 0.5:
                    inactive.append(mol)

        print("Found {} / {} molecules.".format(len(active), len(inactive)))
        n = min(len(active), len(inactive))

        mols = active[:n] + inactive[:n]

        # Predict single molecule rotation
        axis = "x"
        print("Rotating along {}...".format(axis))
        example_results = []
        example_results2 = []
        for active_mol in mols:
            inputs_mol, inputs_cor = [], []
            for degree in np.arange(0, 360, 5):
                mol = rotate_molecule(active_mol, axis, degree)
                inputs_mol.append(mol)
                inputs_cor.append(mol.GetConformer().GetPositions())
            gen = MPGenerator(inputs_mol, inputs_cor, [1] * len(inputs_mol), 16, task="input_only",
                              num_atoms=int(hyper["num_atoms"]))
            example_results.append(trainer.model.predict_generator(gen).flatten())
            example_results2.append(trainer2.model.predict_generator(gen).flatten())

        with open("../../experiment/figure/rotation_single_bace_{}_x.csv".format(i), "w") as file:
            writer = csv.writer(file, delimiter=",")
            for row in example_results:
                writer.writerow(row)

        with open("../../experiment/figure/rotation_single_bacer_{}_x.csv".format(i), "w") as file:
            writer = csv.writer(file, delimiter=",")
            for row in example_results2:
                writer.writerow(row)


def rotate_molecule(mol, axis, degree):
    rotation_matrix = degree_rotation_matrix(axis, float(degree))
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
        pos_rotated = np.matmul(rotation_matrix, pos)
        mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
    return mol


def draw_bar_graph(path):
    results = []
    dataset = path.split("/")[4]
    for axis in ["x", "y", "z"]:
        # Rotation - random
        with open(path + "/rotation_random.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([*mean, *std])

        # Rotation - stepwise (negative)
        for degree in [225, 270, 315]:
            with open(path + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

        # Control
        with open(path + "/results.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([mean[2], mean[5], std[2], std[5]])

        # Rotation - stepwise (positive)
        for degree in [45, 90, 135, 180]:
            with open(path + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

    results_x, results_y, results_z = np.split(np.array(results), 3)

    # Draw figure
    # x, y, z rotation
    plt.figure(figsize=(8, 2.5))
    ticks = np.arange(9)
    plt.errorbar(ticks, results_x[:, 0], results_x[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                 color="#000000", linestyle='solid')
    plt.errorbar(ticks, results_y[:, 0], results_y[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                 color="#000000", linestyle='dashed')
    plt.errorbar(ticks, results_z[:, 0], results_z[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                 color="#000000", linestyle='dotted')
    plt.xticks(ticks, ("Rand", "-135", "-90", "-45", "0", "+45", "+90", "+135", "+180"))

    if dataset == "bace_cla" or dataset == "bace_rotated":
        plt.ylim(top=1.0, bottom=0.6)
        plt.yticks(np.arange(0.6, 1.01, 0.1))
    elif dataset == "hiv":
        plt.ylim(top=1.0, bottom=0.6)
        plt.yticks(np.arange(0.6, 1.01, 0.1))
    elif dataset == "delaney":
        plt.ylim(top=1.0, bottom=0.0)
        plt.yticks(np.arange(0.0, 1.001, 0.2))
    elif dataset == "freesolv":
        plt.ylim(top=1.0, bottom=0.0)
        plt.yticks(np.arange(0.0, 1.001, 0.2))

    fig_name = "./{}_rotation_graph.png".format(dataset)
    plt.savefig(fig_name, dpi=600)


def draw_bar_graph_adj(path1, path2):
    results = []
    for axis in ["x", "y", "z"]:
        # Rotation - random
        with open(path1 + "/rotation_random.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([*mean, *std])

        # Rotation - stepwise (negative)
        for degree in [225, 270, 315]:
            with open(path1 + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

        # Control
        with open(path1 + "/results.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([mean[2], mean[5], std[2], std[5]])

        # Rotation - stepwise (positive)
        for degree in [45, 90, 135, 180]:
            with open(path1 + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

    results_x, results_y, results_z = np.split(np.array(results), 3)

    results = []
    for axis in ["x", "y", "z"]:
        # Rotation - random
        with open(path2 + "/rotation_random.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([*mean, *std])

        # Rotation - stepwise (negative)
        for degree in [225, 270, 315]:
            with open(path2 + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

        # Control
        with open(path2 + "/results.csv") as file:
            reader = csv.reader(file)
            next(reader)
            mean = np.array(list(next(reader)), dtype=float)
            std = np.array(list(next(reader)), dtype=float)
            results.append([mean[2], mean[5], std[2], std[5]])

        # Rotation - stepwise (positive)
        for degree in [45, 90, 135, 180]:
            with open(path2 + "/rotation_raw_stepwise" + axis + str(degree) + ".csv") as file:
                reader = csv.reader(file)
                next(reader)
                data = np.array(list(reader), dtype=float)
                results.append([*np.array(data).mean(axis=0), *np.array(data).std(axis=0)])

    results_x2, results_y2, results_z2 = np.split(np.array(results), 3)

    # Draw figure
    # x, y, z rotation
    # w, h = plt.figaspect(0.33)
    # plt.figure(figsize=(8, 1.25))
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3.5))
    fig.subplots_adjust(hspace=0)

    ticks = np.arange(9)
    ax[0].errorbar(ticks, results_x[:, 0], results_x[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='solid')
    ax[0].errorbar(ticks, results_y[:, 0], results_y[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='dashed')
    ax[0].errorbar(ticks, results_z[:, 0], results_z[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='dotted')
    # ax[0].set_xticks(ticks, ("Rand", "-135", "-90", "-45", "0", "+45", "+90", "+135", "+180"))

    ax[0].set_ylim(top=0.95, bottom=0.65)
    ax[0].set_yticks(np.arange(0.70, 1.01, 0.2))

    ax[1].errorbar(ticks, results_x2[:, 0], results_x2[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='solid')
    ax[1].errorbar(ticks, results_y2[:, 0], results_y2[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='dashed')
    ax[1].errorbar(ticks, results_z2[:, 0], results_z2[:, 2], capsize=4, elinewidth=2, markeredgewidth=1,
                   color="#000000", linestyle='dotted')
    ax[1].set_xticks(ticks, ("Rand", "-135", "-90", "-45", "0", "+45", "+90", "+135", "+180"))

    ax[1].set_ylim(top=0.95, bottom=0.65)
    ax[1].set_yticks(np.arange(0.70, 1.01, 0.2))

    fig_name = "./bace_and_rotated_rotation_graph.png"
    plt.savefig(fig_name, dpi=600)


def draw_example_graph(dataset, trial_path):
    results = []
    with open(trial_path + "/rotation_single_x.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            result = [float(r) for r in row]  # ex [0, 45, 90, 135, 180, 225, 270, 315]
            results.append([*result[len(result) // 2:], *result[:len(result) // 2 + 1]])

    major_tick = MultipleLocator(18)
    major_formatter = FixedFormatter(["", "-180", "-90", "0", "+90", "+180"])
    minor_tick = MultipleLocator(9)

    x = np.arange(len(results[0]))

    # Draw figure
    for j in range(0, min(len(results), 5)):
        if "bace" in trial_path or "hiv" in trial_path:
            plt.figure(figsize=(8, 2.5))
            ax = plt.subplot(1, 1, 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.plot(x, results[j], color="#000000", linewidth=2)

            # Left ticks
            ax.xaxis.set_major_locator(major_tick)
            ax.xaxis.set_major_formatter(major_formatter)
            ax.xaxis.set_minor_locator(minor_tick)
            ax.xaxis.set_minor_formatter(NullFormatter())
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1.01, 0.5), ("0.0", "0.5", "1.0"))

            fig_name = "../../experiment/figure/ex/rotation_single_{}_{}_x.png".format(dataset, j)
            plt.savefig(fig_name, dpi=600)
            plt.clf()
            print("Saved figure on {}".format(fig_name))

        else:
            # Figure
            plt.figure(figsize=(8, 2.5))
            ax = plt.subplot(1, 1, 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            y = results[j]
            mean_y = np.average(y)
            ylim = (mean_y - 1.5, mean_y + 1.5)
            plt.plot(x, y, color="#000000", linewidth=2)

            # Ticks
            ax.xaxis.set_major_locator(major_tick)
            ax.xaxis.set_major_formatter(major_formatter)
            ax.xaxis.set_minor_locator(minor_tick)
            ax.xaxis.set_minor_formatter(NullFormatter())
            plt.ylim(ylim)

            fig_name = "../../experiment/figure/ex/rotation_single_{}_{}_x.png".format(dataset, j)
            plt.savefig(fig_name, dpi=600)
            plt.clf()
            print("Saved figure on {}".format(fig_name))


def draw_bace_example_graph():
    # active/inactive example
    for i in range(8, 11):
        results = []
        with open("../../experiment/figure/rotation_single_bace_{}_x.csv".format(i)) as file:
            reader = csv.reader(file)
            for row in reader:
                result = [float(r) for r in row]  # ex [0, 45, 90, 135, 180, 225, 270, 315]
                results.append([*result[len(result) // 2:], *result[:len(result) // 2 + 1]])

        active_results = results[:len(results) // 2]
        inactive_results = results[len(results) // 2:]

        print(active_results)

        results = []
        with open("../../experiment/figure/rotation_single_bacer_{}_x.csv".format(i)) as file:
            reader = csv.reader(file)
            for row in reader:
                result = [float(r) for r in row]  # ex [0, 45, 90, 135, 180, 225, 270, 315]
                results.append([*result[len(result) // 2:], *result[:len(result) // 2 + 1]])

        active_results2 = results[:len(results) // 2]
        inactive_results2 = results[len(results) // 2:]

        print(active_results2)

        major_tick = MultipleLocator(9)
        major_formatter = FixedFormatter(["", "-180", "-135", "-90", "-45", "0", "+45", "+90", "+135", "+180"])
        minor_tick = MultipleLocator(9)

        x = np.arange(len(active_results[0]))

        for j in range(len(active_results)):
            # plt.figure(figsize=(8, 2.5))
            plt.figure(figsize=(17, 2.5))
            ax = plt.subplot(1, 1, 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Active
            plt.plot(x, active_results[j], color="#000000", linewidth=2, linestyle="solid")
            plt.plot(x, active_results2[j], color="#000000", linewidth=2, linestyle="dashed")

            # Left ticks
            ax.xaxis.set_major_locator(major_tick)
            ax.xaxis.set_major_formatter(major_formatter)
            ax.xaxis.set_minor_locator(minor_tick)
            ax.xaxis.set_minor_formatter(NullFormatter())
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1.01, 0.5), ("0.0", "0.5", "1.0"))

            fig_name = "../../experiment/figure/ex/rotation_single_trial{}_a{}_x.png".format(i, j)
            plt.savefig(fig_name, dpi=600)
            plt.clf()
            print("Saved figure on {}".format(fig_name))

            # plt.figure(figsize=(8, 2.5))
            plt.figure(figsize=(17, 2.5))
            ax = plt.subplot(1, 1, 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Inactive
            plt.plot(x, inactive_results[j], color="#000000", linewidth=2, linestyle="solid")
            plt.plot(x, inactive_results2[j], color="#000000", linewidth=2, linestyle="dashed")

            # Left ticks
            ax.xaxis.set_major_locator(major_tick)
            ax.xaxis.set_major_formatter(major_formatter)
            ax.xaxis.set_minor_locator(minor_tick)
            ax.xaxis.set_minor_formatter(NullFormatter())
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1.01, 0.5), ("0.0", "0.5", "1.0"))

            fig_name = "../../experiment/figure/ex/rotation_single_trial{}_i{}_x.png".format(i, j)
            plt.savefig(fig_name, dpi=600)
            plt.clf()
            print("Saved figure on {}".format(fig_name))


if __name__ == "__main__":
    rotation_prediction("../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017")
    rotation_prediction("../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017")
    rotation_prediction("../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018")
    rotation_prediction("../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113")
    rotation_prediction("../../result/model_3DGCN/hiv/16_c128_d128_l2_psum_022612")

    for axis in ["x", "y", "z"]:
        for deg in [45, 90, 135, 180, 225, 270, 315]:
            rotation_prediction("../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017", rotation="stepwise",
                                axis=axis, degree=deg)

    for axis in ["x", "y", "z"]:
        for deg in [45, 90, 135, 180, 225, 270, 315]:
            rotation_prediction("../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017", rotation="stepwise",
                                axis=axis, degree=deg)

    for axis in ["x", "y", "z"]:
        for deg in [45, 90, 135, 180, 225, 270, 315]:
            rotation_prediction("../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113",
                                rotation="stepwise", axis=axis, degree=deg)

    for axis in ["x", "y", "z"]:
        for deg in [45, 90, 135, 180, 225, 270, 315]:
            rotation_prediction("../../result/model_3DGCN/hiv/16_c128_d128_l2_psum_022612",
                                rotation="stepwise", axis=axis, degree=deg)

    single_molecule_rotation("../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017")
    single_molecule_rotation("../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017")
    single_molecule_rotation("../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018")
    single_molecule_rotation("../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113")
    single_molecule_rotation("../../result/model_3DGCN/hiv/16_c128_d128_l2_psum_022612")

    draw_bar_graph("../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017")
    draw_bar_graph("../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017")
    draw_bar_graph("../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018")
    draw_bar_graph("../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113")
    draw_bar_graph("../../result/model_3DGCN/hiv/16_c128_d128_l2_psum_022612")

    draw_example_graph("delaney", "../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017/trial_7")
    draw_example_graph("freesolv", "../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017/trial_10")
    draw_example_graph("bace_cla", "../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018/trial_4")
    draw_example_graph("bace_rotated", "../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113/trial_9")
    draw_example_graph("hiv", "../../result/model_3DGCN/hiv/16_c128_d128_l2_psum_022612/trial_5")

    draw_bar_graph_adj("../../result/model_3DGCN/bace_cla/16_c128_d128_l2_pmax_022018",
                       "../../result/model_3DGCN/bace_rotated/16_c128_d128_l2_pmax_022113")

    bace_overlap_single_rotation()
    draw_bace_example_graph()
