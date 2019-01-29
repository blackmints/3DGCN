from keras.utils import Sequence, to_categorical
from rdkit.Chem import rdmolops, rdchem, AllChem
from model.features import *
import numpy as np


class Dataset(object):
    def __init__(self, dataset, batch=128, normalize=False, rotate=0):
        self.dataset = dataset
        self.data_format = None
        self.batch = batch
        self.normalize = normalize
        self.outputs = 1
        self.smiles = []
        self.coords = []
        self.target = []
        self.x, self.c, self.y = {}, {}, {}

        # Load data
        self.load_dataset(dataset, rotate=rotate)

        # Get dataset parameters
        self.num_atoms, self.num_features = self.get_parameters()

        # Normalize
        if self.task == "regression" and normalize:
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])

            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std
        else:
            self.mean = 0
            self.std = 1

    def load_file(self, path, target_name="target"):
        x, y, c = [], [], []

        self.data_format = "mol"
        mols = Chem.SDMolSupplier(path)

        for mol in mols:
            if mol is not None:
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                    self.outputs = len(target_name)

                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    continue

                x.append(mol)
                c.append(mol.GetConformer().GetPositions())


        assert len(x) == len(c) == len(y)
        return x, c, y

    def load_dataset(self, dataset, rotate=0):
        if dataset == "delaney":
            path, max_atoms, max_number, target_name = "../data/delaney/delaney.sdf", 0, 0, "target"
            self.task = "regression"
        elif dataset == "bace_cla":
            path, max_atoms, max_number, target_name = "../data/bace/bace.sdf", 0, 0, "Class"
            self.task = "binary"
        elif dataset == "bace_reg":
            path, max_atoms, max_number, target_name = "../data/bace/bace.sdf", 0, 0, "pIC50"
            self.task = "regression"
        elif dataset == "freesolv":
            path, max_atoms, max_number, target_name = "../data/freesolv/freesolv.sdf", 0, 0, "exp"
            self.task = "regression"
        else:
            path, max_atoms, max_number, target_name = "", 0, 0, "target"
            assert dataset is not None, 'Unsupported dataset: {}'.format(dataset)

        x, c, y = self.load_file(path, target_name=target_name)

        # Filter with maximum number of atoms
        new_smiles, new_coords, new_target = [], [], []
        if max_atoms > 0:
            for mol, coor, tar in zip(x, c, y):
                num_atoms = mol.GetNumAtoms()

                if num_atoms <= max_atoms:
                    new_smiles.append(mol)
                    new_coords.append(coor)
                    new_target.append(tar)

            x = new_smiles
            c = new_coords
            y = new_target

        if self.task != "regression":
            self.smiles, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.smiles, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        idx = np.random.permutation(len(self.smiles))
        self.smiles, self.coords, self.target = self.smiles[idx], self.coords[idx], self.target[idx]

        # Cut with maximum number of data
        if max_number > 0:
            self.smiles, self.coords, self.target = self.smiles[:max_number], \
                                                    self.coords[:max_number], \
                                                    self.target[:max_number]

        # Split data
        spl1 = int((len(self.smiles) // (rotate + 1)) * 0.2) * (rotate + 1)
        spl2 = int((len(self.smiles) // (rotate + 1)) * 0.1) * (rotate + 1)

        self.x = {"train": self.smiles[spl1:],
                  "valid": self.smiles[spl2:spl1],
                  "test": self.smiles[:spl2]}
        self.c = {"train": self.coords[spl1:],
                  "valid": self.coords[spl2:spl1],
                  "test": self.coords[:spl2]}
        self.y = {"train": self.target[spl1:],
                  "valid": self.target[spl2:spl1],
                  "test": self.target[:spl2]}
        assert len(self.x["train"]) == len(self.y["train"]) == len(self.c["train"])
        assert len(self.x["valid"]) == len(self.y["valid"]) == len(self.c["valid"])
        assert len(self.x["test"]) == len(self.y["test"]) == len(self.c["test"])

    def save_dataset(self, pred, path, target="test", target_path=None):
        mols = []
        for idx, (x, c, y, p) in enumerate(zip(self.x[target], self.c[target], self.y[target], pred)):
            if self.data_format == "smiles":
                mol = Chem.MolFromSmiles(x)
                AllChem.EmbedMolecule(mol)

                # Set coordinates
                for atom in mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    mol.GetConformer().SetAtomPosition(atom_idx, c[atom_idx])
            else:
                mol = x

            mol.SetProp("true", str(y * self.std + self.mean))
            mol.SetProp("pred", str(p[0] * self.std + self.mean))
            mols.append(mol)

        w = Chem.SDWriter(path + target + "_results.sdf" if target_path is None else target_path)
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def get_parameters(self):
        n_atom_features = num_atom_features()
        n_atoms = 0

        for mol_idx, mol in enumerate(self.smiles):
            if type(mol) is not rdchem.Mol:
                mol = Chem.MolFromSmiles(mol)

            if mol is not None:
                n_atoms = max(n_atoms, mol.GetNumAtoms())

        return n_atoms, n_atom_features

    def generator(self, target, task=None):
        return MPGenerator(self.x[target], self.c[target], self.y[target], self.batch,
                           task=task if task is not None else self.task,
                           num_atoms=self.num_atoms, num_features=self.num_features)


class MPGenerator(Sequence):
    def __init__(self, x_set, c_set, y_set, batch, task="binary", num_atoms=0, num_features=0):
        assert len(x_set) == len(c_set) == len(y_set)
        self.x, self.c, self.y = x_set, c_set, y_set

        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms
        self.num_features = num_features

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        if self.task == "category":
            return self._tensorize(batch_x, batch_c), to_categorical(batch_y)
        elif self.task == "binary":
            return self._tensorize(batch_x, batch_c), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self._tensorize(batch_x, batch_c), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self._tensorize(batch_x, batch_c)

    def _tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_features))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))

        for mol_idx, mol in enumerate(batch_x):
            atoms = mol.GetAtoms()

            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                atom_tensor[mol_idx, atom_idx, :] = atom_features(atom)

                # 3D Coordinate
                pos_c = batch_c[mol_idx][atom_idx]

                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    pos_n = batch_c[mol_idx][neighbor_idx]
                    # Direction should be Neighbor -> Center
                    c_to_n = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = c_to_n

            # Adjacency matrix
            adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")
            adjm_tensor[mol_idx, : len(atoms), : len(atoms)] = adjms

        return [atom_tensor, adjm_tensor, posn_tensor]
