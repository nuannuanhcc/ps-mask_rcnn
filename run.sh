export DIR="./train_log/faster_sysu_res101"
export NGPUS=1
export LR=$(echo "$NGPUS*0.001"|bc)
export BATCH=$(echo "$NGPUS*4"|bc)

python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
--config-file "./configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" \
 OUTPUT_DIR $DIR \
 SOLVER.IMS_PER_BATCH $BATCH \
 SOLVER.BASE_LR $LR \
 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 8000

 # e2e_faster_rcnn_R_101_FPN_1x.yaml
 # e2e_faster_rcnn_R_50_FPN_1x.yaml
 # e2e_faster_rcnn_R_50_C4_1x.yaml
 # retinanet/retinanet_R-101-FPN_P5_1x.yaml
 # retinanet/retinanet_R-50-FPN_P5_1x.yaml