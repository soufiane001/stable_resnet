dataset=cifar100 # cifar10

train_depth () {
    local depth=$1
    local config=configs/$dataset/resnet$depth/config.json
    for init_lr in 0.05 0.1 0.5
    do
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 7 &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 6 --scaling none &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 4 --bn &
        python3 main.py --config $config --init_lr $init_lr --n_runs 3 --gpu 5 --bn --scaling dec
    done
}

for depth in 32 50
do 
    train_depth $depth
done
