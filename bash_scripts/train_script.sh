train_depth () {
    local depth=$1
    local dataset=$2
    local config=configs/$dataset/resnet$depth/config.json
    for init_lr in 0.02 0.1 0.5
    do
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 0 --scaling None &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 1 --scaling Uniform &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 3 --scaling Decrease &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 4 --bn --scaling None &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 6 --bn --scaling Decrease &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 7 --bn --scaling Uniform
        wait
    done
}
for dataset in cifar100
do
    for depth in 32 50 104
    do
        train_depth $depth $dataset
    done
done
