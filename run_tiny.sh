NUM_PROC=4
#CONFIG_FILE=configs/swin_tiny_patch4_window7_224.yaml
CONFIG_FILE=configs/swin_tiny_patch2_window4_cifar.yaml
#RESUME_FILE=checkpoints/swin_tiny_patch4_window7_224.pth
RESUME_FILE=checkpoints/swin_tiny_patch2_window4_cifar.pth
#DATA_PATH=../datasets/imagenet/
DATA_PATH=../datasets/cifar10/

CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=$NUM_PROC python -m torch.distributed.launch --nproc_per_node $NUM_PROC --master_port 12345 \
    main.py --cfg $CONFIG_FILE \
    --data-path $DATA_PATH \
#    --resume $RESUME_FILE
