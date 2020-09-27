train_depth () {
    local depth=$1
    local dataset=$2
    local config=configs/$dataset/resnet$depth/config.json
    # for init_lr in 0.1
    # do
    # python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 0 --scaling None &
    # python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 1 --scaling Uniform &
    # python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 3 --scaling Decrease &
    python3 main.py --config $config --init_lr 0.1 --n_runs 3 --gpu 4 --bn True --scaling None &
    python3 main.py --config $config --init_lr 0.1 --n_runs 3 --gpu 6 --bn True --scaling Decrease &
    python3 main.py --config $config --init_lr 0.1 --n_runs 3 --gpu 7 --bn True --scaling Uniform
    wait
    # done
}
for dataset in tiny_imagenet
do
    for depth in 104
    do
        train_depth $depth $dataset
    done
done
