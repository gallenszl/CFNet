# CFNet(CVPR 2021)
This is the implementation of the paper CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching, `CVPR 2021`, Zhelun Shen, Yuchao Dai, Zhibo Rao [\[Arxiv\]](https://arxiv.org/abs/2104.04314).

Our method also obtains the `1st` place on the stereo task of `Robust Vision Challenge 2020`

Camera ready version and supplementary Materials can be found in [\[CVPR official website\]](https://openaccess.thecvf.com/content/CVPR2021/html/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.html)

## News
Our extended journal articles have been accepted by **TPAMI**. Please see [\[this website\]](https://github.com/gallenszl/UCFNet/tree/main) for more details.

## Abstract
Recently, the ever-increasing capacity of large-scale annotated datasets has led to profound progress in stereo matching. However, most of these successes are limited to a specific dataset and cannot generalize well to other datasets. The main difficulties lie in the large domain differences and unbalanced disparity distribution across a variety of datasets, which greatly limit the real-world applicability of current deep stereo matching models. In this paper, we propose CFNet, a Cascade and Fused cost volume based network to improve the robustness of the stereo matching network. First, we propose a fused cost volume representation to deal with the large domain difference. By fusing multiple low-resolution dense cost volumes to enlarge the receptive field, we can extract robust structural representations for initial disparity estimation. Second, we propose a cascade cost volume representation to alleviate the unbalanced disparity distribution. Specifically, we employ a variance-based uncertainty estimation to adaptively adjust the next stage disparity search space, in this way driving the network progressively prune out the space of unlikely correspondences. By iteratively narrowing down the disparity search space and improving the cost volume resolution, the disparity estimation is gradually refined in a coarse-tofine manner. When trained on the same training images and evaluated on KITTI, ETH3D, and Middlebury datasets with the fixed model parameters and hyperparameters, our proposed method achieves the state-of-the-art overall performance and obtains the 1st place on the stereo task of Robust Vision Challenge 2020.

# How to use

## Environment
* python 3.74
* Pytorch == 1.1.0
* Numpy == 1.15

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), [ETH3D](https://www.eth3d.net/), [Middlebury](https://vision.middlebury.edu/stereo/)


**KITTI2015/2012 SceneFlow**

please place the dataset as described in `"./filenames"`, i.e., `"./filenames/sceneflow_train.txt"`, `"./filenames/sceneflow_test.txt"`, `"./filenames/kitticombine.txt"`

**Middlebury/ETH3D**

Our folder structure is as follows:
```
dataset
├── KITTI2015
├── KITTI2012
├── Middlebury
    │ ├── Adirondack
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
├── ETH3D
    │ ├── delivery_area_1l
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
```
Note that we use the full-resolution image of Middlebury for training as the additional training images don't have half-resolution version. We will down-sample the input image to half-resolution in the data argumentation. In contrast,  we use the half-resolution image and full-resolution disparity of Middlebury for testing. 

## Training
**Scene Flow Datasets Pretraining**

run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.

To repeat our pretraining details. You may need to replace the Mish activation function to Relu. Samples is shown in `./models/relu/`.

**Finetuning**

run the script `./scripts/robust.sh` to jointly finetune the pre-train model on four datasets,
i.e., KITTI 2015, KITTI2012, ETH3D, and Middlebury. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.

## Evaluation
**Joint Generalization**

run the script `./scripts/eth3d_save.sh"`, `./scripts/mid_save.sh"` and `./scripts/kitti15_save.sh` to save png predictions on the test set of the ETH3D, Middlebury, and KITTI2015 datasets. Note that you may need to update the storage path of save_disp.py, i.e., `fn = os.path.join("/home3/raozhibo/jack/shenzhelun/cfnet/pre_picture/"`, fn.split('/')[-2]).

**Corss-domain Generalization**

run the script `./scripts/robust_test.sh"` to test the cross-domain generalizaiton of the model (Table.3 of the main paper). Please update `--loadckpt` as pretrained SceneFlow checkpoint file.

## Pretrained Models

[Pretraining Model](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in Table.3 of the main paper

[Finetuneing Moel](https://drive.google.com/file/d/1H6L-lQjF4yOxq23wxs3HW40B-0mLxUiI/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in the stereo task of Robust Vision Challenge 2020

## Citation
If you find this code useful in your research, please cite:
```
@InProceedings{Shen_2021_CVPR,
    author    = {Shen, Zhelun and Dai, Yuchao and Rao, Zhibo},
    title     = {CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13906-13915}
}
```
# Acknowledgements

Thanks to the excellent work GWCNet, Deeppruner, and HSMNet. Our work is inspired by these work and part of codes are migrated from [GWCNet](https://github.com/xy-guo/GwcNet), [DeepPruner](https://github.com/uber-research/DeepPruner/) and [HSMNet](https://github.com/gengshan-y/high-res-stereo).
