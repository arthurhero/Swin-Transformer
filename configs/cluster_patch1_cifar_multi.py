MODEL:
  TYPE: cluster
  NAME: cluster_patch1_cifar_multi
  DROP_PATH_RATE: 0.2
  SWIN:
    EMBED_DIM: 96
    DEPTHS: [ 2, 2, 8 ]
    NUM_HEADS: [ 3, 6, 12 ]
    PATCH_SIZE: 1
  CLUSTER:
    POS_DIM = 2
    K = [64,16,4,1]
    POS_LAMBDA = [0.0003,0.0001,0.00003,0]
DATA:
  DATASET: cifar
  IMG_SIZE: 32
  BATCH_SIZE: 256
TRAIN:
  EPOCHS: 1000
