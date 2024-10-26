# Minimum Version of ViDiT-Q

## installation

- pip install prequisites `torch, torchvision, timm, omegaconf`
- go to the `./quant_utils/`, run `pip install -e .` (local editable installation of qdiff package)

## Examples

in `./scrpits/dit/`, run `ptq.sh`, then `quant_inference.sh`

the `get_calib_data.py` is used to generate the channel-wise activations

the `main.sh` describes the whole process.

the `sweep_alpha.py` calls the main.sh for search of smooth_quant alpha
