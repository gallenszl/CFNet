# CFNet
This is the implementation of the paper CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching, CVPR 2021, Zhelun Shen, Yuchao Dai, Zhibo Rao [\[Arxiv\]](https://arxiv.org/abs/2104.04314). 

Camera ready version and supplementary Materials can be found in [\[CVPR official website\]](https://openaccess.thecvf.com/content/CVPR2021/html/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.html)

# Code will be available before July 2021.

# Thank you for your concerns about our work. I am really sorry that we may delay one or two weeks to publish our code.  

## Abstract
Recently, the ever-increasing capacity of large-scale annotated datasets has led to profound progress in stereo matching. However, most of these successes are limited to a specific dataset and cannot generalize well to other datasets. The main difficulties lie in the large domain differences and unbalanced disparity distribution across a variety of datasets, which greatly limit the real-world applicability of current deep stereo matching models. In this paper, we propose CFNet, a Cascade and Fused cost volume based network to improve the robustness of the stereo matching network. First, we propose a fused cost volume representation to deal with the large domain difference. By fusing multiple low-resolution dense cost volumes to enlarge the receptive field, we can extract robust structural representations for initial disparity estimation. Second, we propose a cascade cost volume representation to alleviate the unbalanced disparity distribution. Specifically, we employ a variance-based uncertainty estimation to adaptively adjust the next stage disparity search space, in this way driving the network progressively prune out the space of unlikely correspondences. By iteratively narrowing down the disparity search space and improving the cost volume resolution, the disparity estimation is gradually refined in a coarse-tofine manner. When trained on the same training images and evaluated on KITTI, ETH3D, and Middlebury datasets with the fixed model parameters and hyperparameters, our proposed method achieves the state-of-the-art overall performance and obtains the 1st place on the stereo task of Robust Vision Challenge 2020.


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
