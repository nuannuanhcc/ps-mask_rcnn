export DIR="./train_log/4_8_0"
export NGPUS=1
export BATCH=$(echo "$NGPUS*8"|bc)

for i in $(seq 1 18)
do
  let ITER=i*2500
  ITER=`printf "%07d" $ITER`
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py \
  --config-file "./configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
  --ckpt $DIR"/model_"$ITER".pth" \
  OUTPUT_DIR $DIR TEST.IMS_PER_BATCH $BATCH
done