import argparse

from classifier_models.test import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--model_path", type=str, default = "trained_classifiers/notguided_fixed_full_cpu/MIAGAE_QGNN1_lr0.1_batch32_layers6/best_model.pt")
parser.add_argument("--num_kfolds", type=int, default=5, help="Number of k-folds to be used for testing")
parser.add_argument("--compressed", action='store_true', help="Whether the data is compressed or not")
parser.add_argument("--gae_type", type=str, default="MIAGAE", help="Type of autoencoder to use to compress data")
parser.add_argument("--gae_model_path", type=str, default="trained_gaes/fixed_full_cpu/MIAGAE_lr0.01_batch256/", help="Path to the autoencoder model")
parser.add_argument("--compressed_data_path", type=str, default="compressed_data/", help="Path to save the compressed data")

args = parser.parse_args()

args = {
    "data_folder": args.data_folder,
    "model_path": args.model_path,
    "num_kfolds": args.num_kfolds,
    "compressed": args.compressed,
    "gae_type": args.gae_type,
    "gae_model_path": args.gae_model_path,
    "compressed_data_path": args.compressed_data_path
}

main(args)
