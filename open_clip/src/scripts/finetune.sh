export CUDA_VISIBLE_DEVICES=7

torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29510 -m training.main \
    --batch-size 16 \
    --precision amp \
    --workers 4 \
    --save-frequency 1 \
    --dataset-type csv \
    --csv-separator="," \
    --train-data="/data/gckang/clipRT/data/finetune/all_skills_all_train/all_skills_all_train.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-supervision-key supervision \
    --csv-label-key label \
    --warmup 100 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=100 \
    --model="ViT-H-14-378-quickgelu" \
    --pretrained="/data/gckang/clipRT/saved_ckpt/base_ckpt/models--apple--DFN5B-CLIP-ViT-H-14-378/snapshots/ed70445b25e41d9edf778c1a3af7b9a28cb6dfb8/open_clip_pytorch_model.bin"
    #--pretrained="/data/gckang/clipRT/saved_ckpt/ViT-H-378-quickgelu/ViT-H-14-378-quickgelu-oxe-pretrained.pt"
    #--model="ViT-H-14-378-quickgelu" \
    #--pretrained='/data/gckang/clipRT/saved_ckpt/ViT-H-378-quickgelu/ViT-H-14-378-quickgelu-oxe-pretrained.pt'
    #--model="ViT-L-14" \
    #--pretrained="datacomp_xl_s13b_b90k" \
    #--model="ViT-g-14" \
    #--pretrained="laion2b_s34b_b88k"
