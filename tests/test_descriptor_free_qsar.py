import lazyqsar
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
from tdc.single_pred import ADME

data = ADME(
    name='bioavailability_ma',
)
split = data.get_split()
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])

if __name__ == "__main__":
    # descriptor = lazyqsar.qsar.descriptors_dict["morgan"]()
    # descriptor.fit(smiles_train)
    # X = np.array(descriptor.transform(smiles_train))
    # print(f"Descriptors: {X}")
    model = lazyqsar.DescriptorFreeLazyQSAR()
    model.fit(smiles_train, y_train)