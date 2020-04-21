export DIR="./train_log/4_8_0"
export NGPUS=1
export BATCH=$(echo "$NGPUS*8"|bc)

python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py \
--config-file "./configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
 OUTPUT_DIR $DIR \
 TEST.IMS_PER_BATCH $BATCH
 # --ckpt "train_log/model_0020000.pth" \