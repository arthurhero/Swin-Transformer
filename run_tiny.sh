NUM_PROC=1
#CONFIG_FILE=configs/swin_tiny_patch4_window7_224.yaml
#CONFIG_FILE=configs/swin_tiny_patch1_window2_cifar_multi.yaml
#CONFIG_FILE=configs/swin_tiny_patch1_window2_cifar_no_shift.yaml
#CONFIG_FILE=configs/swin_conv_patch1_window2_cifar_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_multi_fix_size.yaml
CONFIG_FILE=configs/cluster_patch1_cifar_equal.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_equal_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_equal_shift_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_equal_shift2_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_layer4_equal_multi.yaml
#CONFIG_FILE=configs/cluster_patch1_cifar_layer6_equal_multi.yaml
#CONFIG_FILE=configs/resnet50_cifar_multi.yaml

#DATA_PATH=../datasets/imagenet/
DATA_PATH=../datasets/cifar10/

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=$NUM_PROC python -m torch.distributed.launch --nproc_per_node $NUM_PROC --master_port 12348 \
    main.py --cfg $CONFIG_FILE \
    --data-path $DATA_PATH \
#    --eval
#    --throughput
