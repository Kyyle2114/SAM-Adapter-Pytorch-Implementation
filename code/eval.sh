#!/bin/bash

TEST_IMAGE_DIR=dataset/test/image
TEST_MASK_DIR=dataset/test/mask

echo SAM_ADAPTER_EVALUATION >> SA_EVAL.txt

python3 eval_sa.py \
    --batch_size 4 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --test_image_dir $TEST_IMAGE_DIR \
    --test_mask_dir $TEST_MASK_DIR \
    >> SA_EVAL.txt
