#!/bin/bash

# Define the input lists
train_data_root="/data/gckang/clipRT/data/train/oxe_scale/"
list_data="fractal20220817_data kuka bridge_orig taco_play jaco_play berkeley_cable_routing roboturk viola berkeley_autolab_ur5 language_table stanford_hydra_dataset_converted_externally_to_rlds austin_buds_dataset_converted_externally_to_rlds nyu_franka_play_dataset_converted_externally_to_rlds furniture_bench_dataset_converted_externally_to_rlds ucsd_kitchen_dataset_converted_externally_to_rlds austin_sailor_dataset_converted_externally_to_rlds austin_sirius_dataset_converted_externally_to_rlds dlr_edan_shared_control_converted_externally_to_rlds iamlab_cmu_pickup_insert_converted_externally_to_rlds utaustin_mutex cmu_stretch bc_z"
list_weight="1.0 0.8341046294 1.0 1.0 1.0 1.0 2.0 1.0 3.0 0.1 1.0 1.0 3.0 0.1 3.0 1.0 1.0 1.0 1.0 1.0 1.0 0.2"


# Initialize empty strings for the concatenated results
all_train_data=""
all_weights=""
all_shards="/*.tar"

# Read the string list into positional parameters
set -- $list_data
for str in "$@"; do
    data_path="${train_data_root}${str}${all_shards}"
    if [ -z "$all_train_data" ]; then
        all_train_data="$data_path"
    else
	all_train_data="${all_train_data}::${data_path}"
    fi
done


# Read the weight list into positional parameters
set -- $list_weight
for num in "$@"; do
    if [ -z "$all_weights" ]; then
        all_weights="$num"
    else
        all_weights="${all_weights}::${num}"
    fi
done

# Print the results
echo "$all_train_data"
echo "$all_val_data"
echo "$all_weights"

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node 4 --rdzv-backend=c10d --rdzv-endpoint=localhost:29510 -m training.main \
    --batch-size 32 \
    --precision amp \
    --workers 4 \
    --save-frequency 1 \
    --train-num-samples 1000000 \
    --dataset-resampled \
    --train-data="$all_train_data" \
    --train-data-upsampling-factors="$all_weights" \
    --dataset-type webdataset \
    --csv-separator="," \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-supervision-key supervision \
    --csv-label-key label \
    --warmup 10000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=100 \
    --model="ViT-H-14-378-quickgelu" \
    --pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--apple--DFN5B-CLIP-ViT-H-14-378/snapshots/ed70445b25e41d9edf778c1a3af7b9a28cb6dfb8/open_clip_pytorch_model.bin"
    #--model="ViT-H-14-quickgelu"
    #--pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--apple--DFN5B-CLIP-ViT-H-14/snapshots/5c8f2637bc994346b98cc52a3f752b8abc632ca7/open_clip_pytorch_model.bin"
    #--model="ViT-bigG-14" \
    #--pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189/open_clip_pytorch_model.bin" \
    #--model="ViT-g-14" \
    #--pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--laion--CLIP-ViT-g-14-laion2B-s34B-b88K/snapshots/15efd0f6ac0c40c0f9da7becca03c974d7012604/open_clip_pytorch_model.bin"
    #--model="ViT-B-16" \
    #--pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots/7288da5a0d6f0b51c4a2b27c624837a9236d0112/open_clip_pytorch_model.bin"
    #--model="ViT-L-14" \
    #--pretrained="laion2b_s32b_b82k"
    #--local-loss \
    #--gather-with-grad \
