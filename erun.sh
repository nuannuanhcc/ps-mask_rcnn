export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py \
--config-file "./configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
 OUTPUT_DIR "./train_log/4_8_1" \
 TEST.IMS_PER_BATCH 16
 # --ckpt "train_log/model_0020000.pth" \