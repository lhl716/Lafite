CONFIG="./experiments/MVTec-LafitE/config_eval.yaml"
GPU=$1
python main/train_val_membank.py \
    -c ${CONFIG} \
    -g ${GPU} \
    -e \
    --coreset_sampling_ratio 0.01 \
    --n_neighbors 3
