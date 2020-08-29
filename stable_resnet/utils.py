import os
import time
import json
import logging
import torch

from pprint import pprint
from easydict import EasyDict as edict


def get_logger(name, logpath, filepath, package_files=[], displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def str_to_list(src, delimiter, converter):
    """Conver a string to list.
    """
    src_split = src.split(delimiter)
    res = [converter(_) for _ in src_split]
    return res


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)

    return config, config_dict


def process_config(json_file):
    """Process a json file into a config file.
    Where we can access the value using .xxx
    Note: we will need to create a similar directory as the config file.
    """
    config, _ = get_config_from_json(json_file)
    paths = json_file.split("/")[1:-1]
    summn = []
    chekn = []
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs"] + paths + summn
    ckpt_dir = ["./runs"] + paths + chekn
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % config.summary_dir)
    print("=> config.checkpoint_dir: %s" % config.checkpoint_dir)
    return config


def get_hypparam_path(scaling, bn, init_lr, schedule):
    path = (
        f'scaling-{scaling}_bn-{bn}_init-lr-{format(init_lr, ".0e")}_sched-{schedule}'
    )
    return path


def find_files(folder, ending=".npy"):
    return list(filter(lambda x: x.endswith(ending), os.listdir(folder)))


def find_subfolders(folder, prefix="scaling"):
    subfolders = [x[0] for x in os.walk(folder)]
    prefix = os.path.join(folder, prefix)
    return list(filter(lambda x: x.startswith(prefix), subfolders))


def strip_suffix(folders, splt_str="_init-lr"):
    return [folder.rsplit(splt_str, 1)[0] for folder in folders]


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def try_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def tensor_to_list(tensor):
    if len(tensor.shape) == 1:
        return [tensor[_].item() for _ in range(tensor.shape[0])]
    else:
        return [tensor_to_list(tensor[_]) for _ in range(tensor.shape[0])]


# =====================================================
# For learning rate schedule
# =====================================================
class StairCaseLRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (
            (start_at >= 0)
            and (iteration >= start_at)
            and (iteration + 1) % interval == 0
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] *= decay_rate
                print("[%d]Decay lr to %f" % (iteration, param_group["lr"]))

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            return lr


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """

    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print("=> Using a preset learning rate schedule:")
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group["lr"])
            param_group["lr"] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            return lr


def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


# =======================================================
# For math computation
# =======================================================
def prod(l):
    val = 1
    if isinstance(l, list):
        for v in l:
            val *= v
    else:
        val = val * l

    return val
