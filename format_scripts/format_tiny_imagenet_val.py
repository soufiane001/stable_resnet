import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

root = "../data/tiny_imagenet"
val_folder = os.path.join(root, "tiny-imagenet-200/val/")

val_dict = {}
with open(os.path.join(val_folder, "val_annotations.txt"), "r") as f:
    for line in f.readlines():
        split_line = line.split("\t")
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(os.path.join(val_folder, "images/*"))
for path in paths:
    file = path.split("/")[-1]
    folder = val_dict[file]
    if not os.path.exists(val_folder + str(folder)):
        os.mkdir(val_folder + str(folder))
        os.mkdir(val_folder + str(folder) + "/images")


for path in paths:
    file = path.split("/")[-1]
    folder = val_dict[file]
    dest = val_folder + str(folder) + "/images/" + str(file)
    move(path, dest)

rmdir(os.path.join(val_folder, "images"))
