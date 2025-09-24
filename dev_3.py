from lazyqsar.preprocess.preprocess import convert_to_onnx

convert_to_onnx("test_dev_2/partition_000")

from lazyqsar.feature_selection.feature_selection_for_binary_classification import convert_to_onnx

convert_to_onnx("test_dev_2/partition_000")

from lazyqsar.latent_variables.latent_variables_for_binary_classification import convert_to_onnx
convert_to_onnx("test_dev_2/partition_000")

from lazyqsar.heads.head_for_binary_classification import convert_to_onnx
convert_to_onnx("test_dev_2/partition_000")