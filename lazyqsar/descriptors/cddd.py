import os
import re
import json
import pandas as pd
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Optional
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from pathlib import Path
from urllib.request import urlretrieve

from ..utils.logging import logger

RDLogger.DisableLog("rdApp.*")

root = os.path.dirname(os.path.abspath(__file__))

REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequence of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)


def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.

    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequence without salts and stereochemistry information.
    """
    try:
        mol = Chem.MolFromSmiles(sml)
        if mol is None:
            return float("nan")

        mol = remover.StripMol(mol, dontRemoveEverything=True)
        if mol is None:
            return float("nan")

        sml = Chem.MolToSmiles(mol, isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        return float("nan")
    return sml


def organic_filter(sml):
    """Function that filters for organic molecules.

    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        if m is None:
            return False
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        return is_organic
    except:
        return False


def filter_smiles(sml):
    """Filter SMILES based on molecular properties.

    Args:
        sml: SMILES sequence.
    Returns:
        Filtered SMILES or nan if outside applicability domain.
    """
    try:
        if isinstance(sml, float) and np.isnan(sml):
            return float("nan")

        m = Chem.MolFromSmiles(sml)
        if m is None:
            return float("nan")

        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET

        if (
            (logp > -5)
            & (logp < 7)
            & (mol_weight > 12)
            & (mol_weight < 600)
            & (num_heavy_atoms > 3)
            & (num_heavy_atoms < 50)
            & is_organic
        ):
            return Chem.MolToSmiles(m)
        else:
            return float("nan")
    except:
        return float("nan")


def preprocess_smiles(sml):
    """Preprocess a SMILES string by removing salts, stereochemistry, and filtering.

    Args:
        sml: SMILES sequence.
    Returns:
        Preprocessed SMILES sequence or nan if invalid/outside domain.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    if isinstance(new_sml, float) and np.isnan(new_sml):
        return float("nan")
    new_sml = filter_smiles(new_sml)
    return new_sml


REGEX_SML = r"Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]"
REGEX_INCHI = r"Br|Cl|[\(\)\+,-/123456789CFHINOPSchpq]"
voc = {
    "#": 1,
    "%": 2,
    "(": 4,
    ")": 3,
    "+": 5,
    "-": 6,
    "0": 8,
    "1": 7,
    "2": 10,
    "3": 9,
    "4": 12,
    "5": 11,
    "6": 14,
    "7": 13,
    "8": 16,
    "9": 15,
    ":": 17,
    "</s>": 0,
    "<s>": 39,
    "=": 18,
    "@": 19,
    "B": 21,
    "Br": 38,
    "C": 20,
    "Cl": 37,
    "F": 22,
    "H": 24,
    "I": 23,
    "N": 26,
    "O": 25,
    "P": 27,
    "S": 28,
    "[": 29,
    "]": 30,
    "c": 31,
    "i": 32,
    "n": 34,
    "o": 33,
    "p": 35,
    "s": 36,
}


class InputPipelineInferEncode:
    """Class that creates a python generator for list of sequnces. Used to feed
    sequnces to the encoing part during inference time.

    Atributes:
        seq_list: List with sequnces to iterate over.
        batch_size: Number of samples to output per iterator call.
        input strings.
    """

    def __init__(self, seq_list, hparams):
        """Constructor for the inference input pipeline class.

        Args:
            seq_list: List with sequnces to iterate over.
            hparams: Hyperparameters for the model, leaving it in the code just for compatibility with the original code, uses just one parameter: batch_size.
        Returns:
            None
        """
        # Preprocess SMILES and filter invalid ones
        processed_smiles = [preprocess_smiles(smi) for smi in seq_list]
        valid_mask = [not pd.isna(smi) for smi in processed_smiles]
        self.seq_list = [
            smi for smi, valid in zip(processed_smiles, valid_mask) if valid
        ]

        if not self.seq_list:
            raise ValueError("No valid SMILES found after preprocessing")

        self.batch_size = hparams.batch_size
        self.encode_vocabulary = voc
        self.generator = None

    def _input_generator(self):
        """Function that defines the generator."""
        l = len(self.seq_list)
        for ndx in range(0, l, self.batch_size):
            samples = self.seq_list[ndx : min(ndx + self.batch_size, l)]
            samples = [self._seq_to_idx(seq) for seq in samples]
            seq_len_batch = np.array([len(entry) for entry in samples])
            # pad sequences to max len and concatenate to one array
            max_length = seq_len_batch.max()
            seq_batch = np.concatenate(
                [
                    np.expand_dims(
                        np.append(
                            seq,
                            np.array(
                                [self.encode_vocabulary["</s>"]]
                                * (max_length - len(seq))
                            ),
                        ),
                        0,
                    )
                    for seq in samples
                ]
            ).astype(np.int32)
            yield seq_batch, seq_len_batch

    def initialize(self):
        """Helper function to initialize the generator"""
        self.generator = self._input_generator()

    def get_next(self):
        """Helper function to get the next batch from the iterator"""
        if self.generator is None:
            self.initialize()
        return next(self.generator)

    def _char_to_idx(self, seq):
        """Helper function to tokenize a sequnce.

        Args:
            seq: Sequence to tokenize.
        Returns:
            List with ids of the tokens in the tokenized sequnce.
        """
        char_list = re.findall(REGEX_SML, seq)
        return [self.encode_vocabulary[char_list[j]] for j in range(len(char_list))]

    def _seq_to_idx(self, seq):
        """Method that tokenizes a sequnce and pads it with start and stop token.

        Args:
            seq: Sequence to tokenize.
        Returns:
            seq: List with ids of the tokens in the tokenized sequnce.
        """
        seq = np.concatenate(
            [
                np.array([self.encode_vocabulary["<s>"]]),
                np.array(self._char_to_idx(seq)).astype(np.int32),
                np.array([self.encode_vocabulary["</s>"]]),
            ]
        ).astype(np.int32)
        return seq


@dataclass
class HParams:
    """Hyperparameters for the model."""

    batch_size: int = 128


class InferenceModel:
    """CDDD Inference Model for encoding SMILES to embeddings and back."""

    def __init__(self):
        """Initialize the inference model."""
        self.hparams = HParams()

        ckpt_dir = Path().home() / ".lazyqsar"
        ckpt_dir.mkdir(exist_ok=True)
        cddd_path = ckpt_dir / "cddd_encoder.onnx"
        if not cddd_path.exists():
            logger.info(
                "Downloading CDDD encoder model into ~/.lazyqsar/cddd_encoder.onnx"
            )
            urlretrieve(
                r"https://zenodo.org/records/14811055/files/encoder.onnx?download=1",
                cddd_path,
            )

        encoder_path = str(cddd_path)
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Model file not found at {encoder_path}. "
                "Please run the model_downloader script first."
            )

        self.encoder_session = ort.InferenceSession(encoder_path)
    
    def seq_to_emb(
        self, smiles_list: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode a list of SMILES strings into molecular descriptors.

        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Optional batch size for inference (default: 128)

        Returns:
            numpy.ndarray: Molecular descriptors for the input SMILES
        """
        if batch_size:
            self.hparams.batch_size = batch_size

        processed_smiles = [preprocess_smiles(smi) for smi in smiles_list]
        good_smiles = []
        accepted_idxs = []
        for i, smiles in enumerate(processed_smiles):
            if str(smiles) == "nan":
                continue
            good_smiles += [smiles]
            accepted_idxs += [i]

        X = np.full((len(smiles_list), 512), np.nan, dtype=np.float32)
        if len(good_smiles) == 0:
            return X

        input_pipeline = InputPipelineInferEncode(good_smiles, self.hparams)
        input_pipeline.initialize()
        emb_list = []
        while True:
            try:
                input_seq, input_len = input_pipeline.get_next()

                outputs = self.encoder_session.run(
                    None,
                    {
                        "Input/Placeholder:0": input_seq.astype(np.int32),
                        "Input/Placeholder_1:0": input_len.astype(np.int32),
                    },
                )
                emb_list.append(outputs[0])
            except StopIteration:
                break
        embeddings = np.vstack(emb_list)
        X[accepted_idxs] = embeddings
        return X


class ContinuousDataDrivenDescriptor(object):
    def __init__(self):
        self.featurizer_name = "cddd"
        self.n_dim = 512
        self.model = InferenceModel()

    def transform(self, smiles_list):
        embeddings = self.model.seq_to_emb(smiles_list)
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings

    def save(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise Exception(f"Directory {dir_name} does not exist.")
        metadata = {
            "featurizer": self.featurizer_name,
            "rdkit_version": Chem.rdBase.rdkitVersion,
        }
        with open(os.path.join(dir_name, "featurizer.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory {dir_name} does not exist.")
        obj = cls()
        with open(os.path.join(dir_name, "featurizer.json"), "r") as f:
            metadata = json.load(f)
            rdkit_version = metadata.get("rdkit_version")
            if rdkit_version:
                logger.debug(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(
                    f"RDKit version mismatch: got {current_rdkit_version}, expected {rdkit_version}"
                )
        return obj
