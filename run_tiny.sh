NUM_PROC=4
CONFIG_FILE=configs/swin_tiny_patch4_window7_224.yaml
#CONFIG_FILE=configs/swin_tiny_patch1_window2_cifar.yaml
#CONFIG_FILE=configs/resnet50_cifar_multi.yaml
#CONFIG_FILE=configs/cluster_imagenet.yaml

DATA_PATH=../datasets/imagenet/

CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=$NUM_PROC python -m torch.distributed.launch --nproc_per_node $NUM_PROC --master_port 12344 \
    main.py --cfg $CONFIG_FILE \
    --data-path $DATA_PATH \
    --batch-size 512 \ # total batch size across all gpu
#    --amp-opt-level O0 \
#    --eval
#    --throughput
