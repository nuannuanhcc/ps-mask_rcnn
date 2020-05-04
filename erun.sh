export DIR="./train_log/4_20_1"
export NGPUS=1
export BATCH=$(echo "$NGPUS*8"|bc)

python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py \
--config-file "./configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
--ckpt $DIR"/model_0045000.pth" \
 OUTPUT_DIR $DIR \
 TEST.IMS_PER_BATCH $BATCH


 # e2e_faster_rcnn_R_50_FPN_1x.yaml
 # e2e_faster_rcnn_R_50_C4_1x.yaml
 # e2e_faster_rcnn_R_50_C5_1x.yaml