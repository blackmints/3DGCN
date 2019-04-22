from rdkit import Chem
import numpy as np

if __name__ == "__main__":
    count = []
    num_mols = 0
    for i in range(1, 11):
        mols = Chem.SDMolSupplier(
            "../../result/model_3DGCN/delaney/8_c128_d128_l2_psum_022017/trial_{}/test.sdf".format(i))
        y = [float(mol.GetProp("true")) - float(mol.GetProp("pred")) for mol in mols]
        y = np.abs(np.array(y))
        num_mols = len(y)
        count.append(len(y[y < 0.30103]))
    print(np.mean(count) / num_mols * 100)

    count = []
    num_mols = 0
    for i in range(1, 11):
        mols = Chem.SDMolSupplier(
            "../../result/model_3DGCN/freesolv/8_c128_d128_l2_psum_022017/trial_{}/test.sdf".format(i))
        y = [float(mol.GetProp("true")) - float(mol.GetProp("pred")) for mol in mols]
        y = np.abs(np.array(y))
        num_mols = len(y)
        count.append(len(y[y < 0.239]))
    print(np.mean(count) / num_mols * 100)
