CONFIG="./experiments/MVTec-LafitE/config.yaml"
GPU=$1
python main/train_val_membank.py \
    -c ${CONFIG} \
    -g ${GPU} \
    --coreset_sampling_ratio 0.01
