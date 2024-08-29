SEEDS=(42 43 44)
DISTORTIONS=(lens_blur white_noise gaussian_blur)
DATASETS=(traffic_inspection factory_inspection zurich_inspection)
for SEED in ${SEEDS[@]}
    do
    python train_ref_model.py --ref-dataset interlaken_inspection --seed $SEED --num-epochs 100
    for DATASET in ${DATASETS[@]}
    do
        for DISTORTION in ${DISTORTIONS[@]}
        do
            python main.py --ref-dataset interlaken_inspection,100,400 --test-dataset $DATASET --distortion-type $DISTORTION \
            --seed $SEED --window 400
        done
    done
done
