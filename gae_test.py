import argparse

from gae_models.test import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--model_path", type=str, default = "trained_gaes/fixed_full_cpu/SAG_model_lr0.001_batch256/best_model.pt")
parser.add_argument("--num_kfolds", type=int, default=5, help="Number of k-folds to be used for cross-validation")

args = parser.parse_args()

args = {
    "data_folder": args.data_folder,
    "model_path": args.model_path,
    "num_kfolds": args.num_kfolds,
}

main(args)
