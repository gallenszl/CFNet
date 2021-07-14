#!/usr/bin/env bash
set -x
DATAPATH="/home1/datasets/Database/robust/kitti12_15/"
CUDA_VISIBLE_DEVICES=0 python save_disp_mid.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet --maxdisp 256 \
--loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/sceneflow_doubletrain/mish45_55/checkpoint_000032.ckpt"