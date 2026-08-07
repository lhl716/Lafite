CONFIG="./experiments/vis_recon/config.yaml"
GPU=$1
for CLASS in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
    echo 'Reconstruct ' ${CLASS}
    python main/vis_recon.py \
        -c ${CONFIG} \
        -g ${GPU} \
        --phase test \
        --class_name ${CLASS}
done
