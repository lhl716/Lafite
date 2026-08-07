CONFIG="./experiments/MVTec-LafitE/config.yaml"
GPU=$1
python main/train_val_diff.py \
    -c ${CONFIG} \
    -g ${GPU}
