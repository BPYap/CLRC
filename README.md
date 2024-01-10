# Group-wise Contrastive Learning

This is the official code repository for the WACV 2024 paper "Group-wise Contrastive Bottleneck for Weakly-Supervised
Visual Representation Learning".

## Environment Setup

```
conda create --name clrc python=3.9
conda activate clrc

conda install cupy cudatoolkit=11.1 -c conda-forge
pip install -r requirements.txt

pip install .
pip install ./SupContrast
```
The `SupContrast` module is adapted from https://github.com/HobbitLong/SupContrast.

## Datasets
Note: Images are not included in this repository. Please refer to the links below to download the images.

| Name            | Link                                                        |
|-----------------|-------------------------------------------------------------|
| UT Zappos 50k   | https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ |
| WIDER Attribtue | http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html    |
| CUB-200-2011    | https://www.vision.caltech.edu/datasets/cub_200_2011/       |
| ImageNet-100    | https://image-net.org/download.php                          |
| Fitzpatrick17k  | https://github.com/mattgroh/fitzpatrick17k                  |

## Pretraining

Configuration files for different datasets and pretrain methods can be found in the `config/pretrain` folder. An example of pretraining on UT Zappos 50k:

```
python main_pretrain.py \
    utzap-attribute \
    clrc \
    --config_file config/pretrain/utzap/clrc.json
```

## Evaluation

Configuration files for different datasets can be found in the `config/downstream` folder. An example of linear evaluation on UT Zappos 50k:

```
python main_finetune.py \
    utzap \
    classifier \
    --backbone_weights model/pretrain/utzap/clrc/lightning_logs/version_0/checkpoints/epoch=999-step=272999.ckpt \
    --config_file config/downstream/utzap.json
```

## Citation

```
@InProceedings{Yap_2024_WACV,
    author    = {Yap, Boon Peng and Ng, Beng Koon},
    title     = {Group-Wise Contrastive Bottleneck for Weakly-Supervised Visual Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2246-2255}
}
```
