NUM_PROC=4
#CONFIG_FILE=configs/swin_tiny_patch4_window7_224.yaml
CONFIG_FILE=configs/swin_tiny_patch1_window2_cifar_multi.yaml
#RESUME_FILE=checkpoints/swin_tiny_patch4_window7_224.pth
#DATA_PATH=../datasets/imagenet/
DATA_PATH=../datasets/cifar10/

CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=$NUM_PROC python -m torch.distributed.launch --nproc_per_node $NUM_PROC --master_port 12346 \
    main.py --cfg $CONFIG_FILE \
    --data-path $DATA_PATH \
    --eval
#    --resume $RESUME_FILE
