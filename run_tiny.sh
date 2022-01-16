python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume checkpoints/swin_tiny_patch4_window7_224.pth --data-path ../datasets/imagenet/ 
