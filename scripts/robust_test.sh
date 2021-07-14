#!/usr/bin/env bash
set -x
DATAPATH="/home1/datasets/Database/robust/kitti12_15/"
CUDA_VISIBLE_DEVICES=0 python robust_test.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_errortest.txt --batch_size 4 --test_batch_size 2 \
    --testlist ./filenames/kitti15_errortest.txt --maxdisp 256 \
    --epochs 1 --lr 0.001  --lrepochs "300:10" \
    --loadckpt "/home3/raozhibo/jack/shenzhelun/unet_confidence_test/down/checkpoints/sceneflow_doubletrain/mish45_55/checkpoint_000032.ckpt" \
    --model cfnet --logdir ./checkpoints/robust_abstudy_test
