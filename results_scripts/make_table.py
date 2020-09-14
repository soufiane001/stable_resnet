import os
from argparse import ArgumentParser

from stable_resnet.results_utils import results_collect, compute_ci_and_format
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", help="What dataset to use", default="cifar100", type=str
    )
    parser.add_argument(
        "--archs", help="Which architectures to use", default="resnet104", nargs="*",
    )
    parser.add_argument(
        "--output_folder",
        default="tables",
        help="Folder in which to save the outputs",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    archs = args.archs if isinstance(args.archs, list) else [args.archs]

    results = []
    for arch in archs:
        results.append(results_collect(args.dataset, arch))
    results = pd.concat(results)
    gb = ["Architecture", "BatchNorm", "Scaled"]
    results = results.groupby(gb).sum()
    results = compute_ci_and_format(results)
    results.index.names = (None, None, None)
    results = results.unstack(level=[-2, -1])
    results.columns = results.columns.droplevel(0)
    results.to_latex(args.output_folder + f"/{args.dataset}_table.tex", escape=False)
