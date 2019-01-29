import numpy as np
from rdkit import Chem


def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_hot(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'H', 'S', 'P', 'Cl', 'Br', 'I', 'B', 'Unknown']) +
                    one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +  # Does not include Hs
                    one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                      Chem.rdchem.HybridizationType.SP2,
                                                      Chem.rdchem.HybridizationType.SP3,
                                                      Chem.rdchem.HybridizationType.SP3D,
                                                      Chem.rdchem.HybridizationType.SP3D2]) +
                    one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +  # Number of implicit Hs
                    [atom.GetIsAromatic()], dtype=int)


def num_atom_features():
    molecule = Chem.MolFromSmiles('CC')
    atom_list = molecule.GetAtoms()

    return len(atom_features(atom_list[0]))
