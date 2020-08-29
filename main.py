import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from stable_resnet.utils import (
    get_logger,
    get_hypparam_path,
    makedirs,
    process_config,
    PresetLRScheduler,
    str_to_list,
    try_cuda,
    is_iterable,
)
from stable_resnet.data_utils import get_dataloader
from stable_resnet.network_utils import get_network


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--scaling",
        type=str,
        help="Scaling to use: [None]/Uniform/Decrease",
        default="None",
    )
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument(
        "--init_lr", help="Initial learning rates", type=float, default=1e-2,
    )
    parser.add_argument(
        "--lr_schedule", help="Learning rate schedule", type=str, default="decay",
    )
    parser.add_argument(
        "--n_runs",
        help="Number of i.i.d. NNs to run (for error bars). Ignored if runs is given.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--runs",
        help="Run id indices (for saving results), taking possibly multiple values",
        default=-1,
        type=int,
        nargs="*",
    )

    args = parser.parse_args()
    config = process_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    return config, args


def init_logger(config, args):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)
    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, "models/base/%s.py" % config.network.lower())
    path_main = os.path.join(path, "main.py")
    logger = get_logger(
        "log",
        logpath=config.summary_dir + "/",
        filepath=path_model,
        package_files=[path_main],
    )
    logger.info(dict(config))
    hypparam_path = get_hypparam_path(
        args.scaling, args.bn, args.init_lr, args.lr_schedule
    )
    summary_writer_path = os.path.join(config.summary_dir, hypparam_path)

    writer = SummaryWriter(summary_writer_path)
    return logger, writer


def train_epoch(net, loader, optimizer, criterion, lr_scheduler, epoch, writer):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)
    desc = "[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
        lr_scheduler.get_lr(optimizer),
        0,
        0,
        correct,
        total,
    )

    writer.add_scalar("train/lr", lr_scheduler.get_lr(optimizer), epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if torch.isnan(loss):
            raise ValueError("Training failed. Loss is NaN.")
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = "[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
            lr_scheduler.get_lr(optimizer),
            train_loss / (batch_idx + 1),
            100.0 * correct / total,
            correct,
            total,
        )
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar("train/loss", train_loss / (batch_idx + 1), epoch)
    writer.add_scalar("train/acc", 100.0 * correct / total, epoch)


def test(net, loader, criterion, epoch, writer):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (test_loss / (0 + 1), 0, correct, total)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                test_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.0 * correct / total

    writer.add_scalar("test/loss", test_loss / (batch_idx + 1), epoch)
    writer.add_scalar("test/acc", 100.0 * correct / total, epoch)
    return acc


def train(
    net,
    trainloader,
    testloader,
    writer,
    config,
    ckpt_path,
    learning_rate,
    weight_decay,
    num_epochs,
    logger,
    args,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )
    if args.lr_schedule == "decay":
        lr_schedule = {
            0: learning_rate,
            int(num_epochs * 0.5): learning_rate * 0.1,
            int(num_epochs * 0.75): learning_rate * 0.01,
        }
    elif args.lr_schedule == "constant":
        lr_schedule = {0: learning_rate}
    else:
        raise ValueError(f"Learning rate schedule: {args.lr_schedule} not found.")
    lr_scheduler = PresetLRScheduler(lr_schedule)
    best_acc = 0
    best_epoch = 0

    test_acc_vec = []
    for epoch in range(num_epochs):
        train_epoch(
            net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer,
        )
        test_acc = test(net, testloader, criterion, epoch, writer)
        test_acc_vec.append(test_acc)
        if test_acc > best_acc:
            print("Saving..")
            best_state = {
                "net": net,
                "acc": test_acc,
                "epoch": epoch,
                "args": config,
            }
            best_acc = test_acc
            best_epoch = epoch
    logger.info("Best acc: %.4f, epoch: %d" % (best_acc, best_epoch))

    return best_state, best_acc, test_acc_vec


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)


def main(config, args):
    # init logger
    logger, writer = init_logger(config, args)

    runs = range(args.n_runs) if args.runs == -1 else args.runs

    weight_decay = config.weight_decay
    num_epochs = config.epochs
    learning_rate = args.init_lr
    # TODO: make this more parallelised
    for run_idx in runs:
        best_acc_vec = []
        test_acc_vec_vec = []
        # build model
        model = get_network(
            config.network,
            config.depth,
            config.dataset,
            use_bn=config.get("use_bn", args.bn),
            scaling=args.scaling,
            act=args.act,
        )

        model = try_cuda(model)
        # preprocessing
        # ====================================== get dataloader ======================================
        trainloader, testloader = get_dataloader(
            config.dataset, config.batch_size, 256, 4
        )
        # ======================================== make paths =========================================
        hypparam_path = get_hypparam_path(
            args.scaling, args.bn, learning_rate, args.lr_schedule
        )
        ckpt_path = os.path.join(config.checkpoint_dir, hypparam_path)
        makedirs(ckpt_path)
        results_path = os.path.join(config.summary_dir, hypparam_path)
        makedirs(results_path)
        # ====================================== fetch exception ======================================
        exception = get_exception_layers(model, str_to_list(config.exception, ",", int))
        logger.info("Exception: ")

        for idx, m in enumerate(exception):
            logger.info("  (%d) %s" % (idx, m))

        # ====================================== fetch training schemes =====================================
        logger.info("Basic Settings: ")
        logger.info(
            " Run %d: LR: %.5f, WD: %.5f, Epochs: %d"
            % (run_idx, learning_rate, weight_decay, num_epochs,)
        )

        best_state, best_acc, test_acc_vec = train(
            net=model,
            trainloader=trainloader,
            testloader=testloader,
            writer=writer,
            config=config,
            ckpt_path=ckpt_path,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            logger=logger,
            args=args,
        )

        best_acc_vec.append(best_acc)
        test_acc_vec_vec.append(test_acc_vec)

        np.save(results_path + f"/run_{run_idx}_best_acc", np.array(best_acc_vec))
        np.save(results_path + f"/run_{run_idx}_test_acc", np.array(test_acc_vec_vec))
        torch.save(best_state, results_path + f"/run_{run_idx}_best_state.pth.tar")


if __name__ == "__main__":
    torch.manual_seed(12)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config, args = init_config()
    main(config, args)
