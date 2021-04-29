# CFNet
This is the implementation of the paper CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching, CVPR 2021, Zhelun Shen, Yuchao Dai, Zhibo Rao [\[Arxiv\]](https://arxiv.org/abs/2104.04314)

# Code will be available before July 2021.

## Abstract
Recently, the ever-increasing capacity of large-scale annotated datasets has led to profound progress in stereo matching. However, most of these successes are limited to a specific dataset and cannot generalize well to other datasets. The main difficulties lie in the large domain differences and unbalanced disparity distribution across a variety of datasets, which greatly limit the real-world applicability of current deep stereo matching models. In this paper, we propose CFNet, a Cascade and Fused cost volume based network to improve the robustness of the stereo matching network. First, we propose a fused cost volume representation to deal with the large domain difference. By fusing multiple low-resolution dense cost volumes to enlarge the receptive field, we can extract robust structural representations for initial disparity estimation. Second, we propose a cascade cost volume representation to alleviate the unbalanced disparity distribution. Specifically, we employ a variance-based uncertainty estimation to adaptively adjust the next stage disparity search space, in this way driving the network progressively prune out the space of unlikely correspondences. By iteratively narrowing down the disparity
search space and improving the cost volume resolution, the disparity estimation is gradually refined in a coarse-tofine manner. When trained on the same training images and evaluated on KITTI, ETH3D, and Middlebury datasets with the fixed model parameters and hyperparameters, our proposed method achieves the state-of-the-art overall performance and obtains the 1st place on the stereo task of Robust Vision Challenge 2020.


## Citation
If you find this code useful in your research, please cite:
```
@inproceedings{cfnet,
title={CFNet: Cascade and Fused Cost Volume for
Robust Stereo Matching},
author={Zhelun Shen and Yuchao Dai and Zhibo
Rao},
booktitle={IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)},
year={2021}
}
```
