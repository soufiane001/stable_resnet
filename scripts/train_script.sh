dataset=cifar10 # cifar100

train_depth () {
    local depth=$1
    local config=configs/$dataset/resnet$depth/config.json
    for init_lr in 0.05 0.1 0.5
    do
        python3 main.py --config $config --init_lr $init_lr --n_runs 2 --gpu 0 &
        python3 main.py --config $config --init_lr $init_lr --n_runs 2 --gpu 1 --scaled &
        python3 main.py --config $config --init_lr $init_lr --n_runs 2 --gpu 2 --bn &
        python3 main.py --config $config --init_lr $init_lr --n_runs 2 --gpu 3 --bn --scaled
    done
}

for depth in 32 50
do 
    train_depth $depth
done