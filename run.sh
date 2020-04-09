export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
--config-file "./configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
 OUTPUT_DIR "./train_log/4_8_1" \
 SOLVER.IMS_PER_BATCH 16
# MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000