CONFIG="./experiments/train_vis_decoder/config.yaml"
GPU=$1
for CLASS in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
    echo 'Train vis_decoder of ' ${CLASS}
    python main/train_vis_decoder.py \
        -c ${CONFIG} \
        -g ${GPU} \
        --class_name ${CLASS}
done
