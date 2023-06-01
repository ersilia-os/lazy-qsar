import os
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import MolFromSmarts

from mordred import Calculator, descriptors

from eosce.models import ErsiliaCompoundEmbeddings

# VARIABLES

PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.abspath(os.path.join(PATH, "..", "data"))


# PROCESSING FUNCTIONS

MAX_NA = 0.2

class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

    def fit(self, X):
        max_na = int((1 - MAX_NA) * X.shape[0])
        idxs = []
        for j in range(X.shape[1]):
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Scaler(object):
    def __init__(self):
        self._name = "scaler"
        self.abs_limit = 10
        self.skip = False

    def set_skip(self):
        self.skip = True

    def fit(self, X):
        if self.skip:
            return
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        if self.skip:
            return X
        X = self.scaler.transform(X)
        X = np.clip(X, -self.abs_limit, self.abs_limit)
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Imputer(object):
    def __init__(self):
        self._name = "imputer"
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class VarianceFilter(object):
    def __init__(self):
        self._name = "variance_filter"

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = self.sel.transform([[i for i in range(X.shape[1])]]).ravel()

    def transform(self, X):
        return self.sel.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


# MORDRED DESCRIPTORS

def mordred_featurizer(smiles):
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas([Chem.MolFromSmiles(smi) for smi in smiles])
    return df


class MordredDescriptor(object):

    def __init__(self):
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()

    def fit(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=self.features)


# CLASSIC DESCRIPTORS

@dataclass
class Descriptors:
    """Molecular descriptors"""

    #: Descriptor type
    descriptor_type: str
    #: Descriptor values
    descriptors: tuple
    # Descriptor name
    descriptor_names: tuple
    # t_stats for each molecule
    tstats: tuple = ()



def _calculate_rdkit_descriptors(mol):
    from rdkit.ML.Descriptors import MoleculeDescriptors  # type: ignore

    dlist = [
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "NumHeteroatoms",
        "RingCount",
        "NumRotatableBonds",
    ]
    c = MoleculeDescriptors.MolecularDescriptorCalculator(dlist)
    d = c.CalcDescriptors(mol)

    def calc_aromatic_bonds(mol):
        return sum(1 for b in mol.GetBonds() if b.GetIsAromatic())

    def _create_smarts(SMARTS):
        s = ",".join("$(" + s + ")" for s in SMARTS)
        _mol = MolFromSmarts("[" + s + "]")
        return _mol

    def calc_acid_groups(mol):
        acid_smarts = (
            "[O;H1]-[C,S,P]=O",
            "[*;-;!$(*~[*;+])]",
            "[NH](S(=O)=O)C(F)(F)F",
            "n1nnnc1",
        )
        pat = _create_smarts(acid_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_basic_groups(mol):
        basic_smarts = (
            "[NH2]-[CX4]",
            "[NH](-[CX4])-[CX4]",
            "N(-[CX4])(-[CX4])-[CX4]",
            "[*;+;!$(*~[*;-])]",
            "N=C-N",
            "N-C=N",
        )
        pat = _create_smarts(basic_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_apol(mol, includeImplicitHs=True):
        # atomic polarizabilities available here:
        # https://github.com/mordred-descriptor/mordred/blob/develop/mordred/data/polarizalibity78.txt

        ap = os.path.join(DATA_PATH, "atom_pols.txt")
        with open(ap, "r") as f:
            atom_pols = [float(x) for x in next(f).split(",")]
        res = 0.0
        for atom in mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum <= len(atom_pols):
                apol = atom_pols[anum]
                if includeImplicitHs:
                    apol += atom_pols[1] * atom.GetTotalNumHs(includeNeighbors=False)
                res += apol
            else:
                raise ValueError(f"atomic number {anum} not found")
        return res

    d = d + (
        calc_aromatic_bonds(mol),
        calc_acid_groups(mol),
        calc_basic_groups(mol),
        calc_apol(mol),
    )
    return d


def classic_featurizer(smiles):
    names = tuple(
        [
            "number of hydrogen bond donor",
            "number of hydrogen bond acceptor",
            "Wildman-Crippen LogP",
            "number of heteroatoms",
            "ring count",
            "number of rotatable bonds",
            "aromatic bonds count",
            "acidic group count",
            "basic group count",
            "atomic polarizability",
        ]
    )
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    R = []
    cols = None
    for m in mols:
        descriptors = _calculate_rdkit_descriptors(m)
        descriptor_names = names
        descriptors = Descriptors(
            descriptor_type="Classic",
            descriptors=descriptors,
            descriptor_names=descriptor_names,
        )
        R += [list(descriptors.descriptors)]
        if cols is None:
            cols = list(descriptors.descriptor_names)
    data = pd.DataFrame(R, columns=cols)
    return data


class ClassicDescriptor(object):

    def __init__(self):
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()

    def fit(self, smiles):
        df = classic_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        df = classic_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=self.features)


# MORGAN FINGERPRINTS

from rdkit.Chem import rdMolDescriptors as rd
from rdkit import Chem

RADIUS = 3
NBITS = 2048
DTYPE = np.uint8

def clip_sparse(vect, nbits):
    l = [0]*nbits
    for i,v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class _MorganDescriptor(object):

    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


def morgan_featurizer(smiles):
    d = _MorganDescriptor()
    X = np.zeros((len(smiles), NBITS))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        X[i,:] = d.calc(mol)
    return X


class MorganDescriptor(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        X = morgan_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = morgan_featurizer(smiles)
        return pd.DataFrame(X, columns=self.features)


# RDKIT 200 Descriptors

from rdkit.Chem import Descriptors as RdkitDescriptors
from rdkit import Chem


RDKIT_PROPS = {"1.0.0": ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
                         'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                         'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                         'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                         'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
                         'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                         'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                         'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',
                         'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                         'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
                         'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
                         'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                         'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                         'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                         'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                         'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
                         'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                         'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                         'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount',
                         'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                         'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                         'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                         'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                         'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                         'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                         'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                         'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                         'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                         'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                         'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                         'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                         'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                         'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                         'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                         'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                         'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                         'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                         'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                         'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                         'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                         'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                         'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
               }

CURRENT_VERSION = "1.0.0"


class _RdkitDescriptor(object):
    def __init__(self):
        self.properties = RDKIT_PROPS[CURRENT_VERSION]
        self._funcs = {name: func for name, func in RdkitDescriptors.descList}

    def calc(self, mols):
        R = []
        for mol in mols:
            if mol is None:
                r = [np.nan]*len(self.properties)
            else:
                r = []
                for prop in self.properties:
                    r += [self._funcs[prop](mol)]
            R += [r]
        return np.array(R)


def rdkit_featurizer(smiles):
    d = _RdkitDescriptor()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    X = d.calc(mols)
    data = pd.DataFrame(X, columns=d.properties)
    return data


class RdkitDescriptor(object):

    def __init__(self):
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()

    def fit(self, smiles):
        df = rdkit_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        df = rdkit_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=self.features)


# MACCS DESCRIPTORS

def maccs_featurizer(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    mk = os.path.join(DATA_PATH, "MACCSkeys.txt")
    with open(str(mk), "r") as f:
        names = tuple([x.strip().split("\t")[-1] for x in f.readlines()[1:]])
    R = []
    cols = None
    for m in mols:
        # rdkit sets fps[0] to 0 and starts keys at 1!
        fps = list(MACCSkeys.GenMACCSKeys(m).ToBitString())[1:] # ersilia edit
        descriptors = tuple(int(i) for i in fps)
        descriptor_names = names
        descriptors = Descriptors(
            descriptor_type="MACCS",
            descriptors=descriptors,
            descriptor_names=descriptor_names,
        )
        R += [list(descriptors.descriptors)]
        if cols is None:
            cols = list(descriptors.descriptor_names)
    data = pd.DataFrame(R, columns=cols)
    return data


class MaccsDescriptor(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        return maccs_featurizer(smiles)

    def transform(self, smiles):
        return maccs_featurizer(smiles)
    
## ERSILIA COMPOUND EMBEDDINGS

class ErsiliaEmbedding(object):
    def __init__(self):
        pass

    def transform(self, smiles):
        mdl = ErsiliaCompoundEmbeddings()
        X = mdl.transform(smiles)
        self.headers = ["ft-{}".format(x) for x in range(1024)]
        return pd.DataFrame(X, columns = self.headers)