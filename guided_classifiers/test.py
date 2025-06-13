# Runs the autoencoder. The normalized or standardized data is imported,
# and the autoencoder model is defined, given the specified options.
# The model is then trained and a loss plot is saved, along with the
# architecture of the model, its hyperparameters, and the best model weights.

import time
import os

from . import util
from .terminal_colors import tcols
from gae_models import data as gae_data
from torch_geometric.loader import DataLoader
from gae_models import util as gae_util
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def main(args):
    device='cpu'
    model_folder = os.path.dirname(args["model_path"])
    hp_gae_file = os.path.join(model_folder, "hyperparameters_gae.json")
    hp_classifier_file = os.path.join(model_folder, "hyperparameters_classifier.json")
    hp_guided_classifier_file = os.path.join(model_folder, "hyperparameters.json")
    hp_gae = gae_util.import_hyperparams(hp_gae_file)
    hp_classifier = gae_util.import_hyperparams(hp_classifier_file)
    hp_guided_classifier = gae_util.import_hyperparams(hp_guided_classifier_file)
    hp = {**hp_gae, **hp_classifier, **hp_guided_classifier}
    
    print(hp)
    # Load the data
    test_graphs = gae_data.SelectGraph(args['data_folder']+"/test")


    y = test_graphs.y

    skf = StratifiedKFold(n_splits=args["num_kfolds"], shuffle=True, random_state=42)

    folds = []

    for _, fold_indices in skf.split(X=range(len(y)), y=y):
        folds.append(fold_indices)


    #test_loader = DataLoader(test_graphs, batch_size=len(test_graphs)//args["num_kfolds"], shuffle=False)

    #Autoencoder model definition
    model = util.choose_guided_classifier_model(hp["gae_type"], hp["classifier_type"], device, hp)

    model.load_model(args["model_path"])
    
    start_time = time.time()

    output_folder = "roc_plots/"
    plot_roc_curve(test_graphs, folds, args["num_kfolds"], model, args["model_path"], output_folder)
          
    end_time = time.time()

    train_time = (end_time - start_time) / 60 

    print(tcols.OKCYAN + f"Testing time: {train_time:.2e} mins." + tcols.ENDC)

def plot_roc_curve(test_graphs, folds_indices, num_kfolds, model, model_path, output_folder):

    #test_loader = DataLoader(test_graphs, batch_size=len(test_graphs)//num_kfolds, shuffle=False)

    roc_aucs, class_outputs, all_tpr, all_fpr = test_kfold_classifier(model,test_graphs, folds_indices, num_kfolds)

    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    mean_fpr = np.linspace(0, 1, 10000)
    tprs_interp = []
    for fpr, tpr in zip(all_fpr, all_tpr):
        # Interpolate the tpr values at the points defined by mean_fpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)

        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs_interp.append(interp_tpr)

    # Compute mean and std across folds (axis=0)

    tprs_interp = np.array(tprs_interp)
    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr  = np.std(tprs_interp, axis=0)

    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)

    np.savez(
        os.path.join(plots_folder, "test_results_data.npz"),
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        std_tpr=std_tpr,
        fold_tprs=tprs_interp,   # shape: (num_folds, len(mean_fpr))
        fold_aucs=roc_aucs,
        mean_roc_auc=mean_roc_auc,
        std_roc_auc=std_roc_auc
    )
    
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)


    fig = plt.figure(figsize=(12, 10))
    plt.plot(mean_fpr, mean_tpr, color="navy", label=f"AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="navy", alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()


    # Save the figure to a file (e.g., PDF or PNG)
    output_file = plots_folder + "roc_curve_with_std.pdf"
    fig.savefig(output_file)

    #fig.savefig(plots_folder + f"roc_plot_mean.pdf")
    np.savetxt(plots_folder + f"fpr_values.txt", fpr, fmt="%f")
    np.savetxt(plots_folder + f"mean_tpr_values.txt", tpr, fmt="%f")

    plt.close()



def test_kfold_classifier(model,test_graphs, folds_indices, num_folds):

    all_losses = []
    all_recon_losses = []
    all_class_losses = []
    all_accuracies = []
    all_roc_aucs = []
    all_class_outputs = []
    all_tprs = []
    all_fprs = []

    for i in range(num_folds):
        test_indices = folds_indices[i]
        test_graphs_fold = test_graphs[test_indices]
        test_loader = DataLoader(test_graphs_fold, batch_size=len(test_graphs_fold), shuffle=False)

        for test_data in test_loader:
            loss, recon_loss, class_loss, class_output = model.compute_loss(test_data)
            loss = loss.item()
            recon_loss = recon_loss.item()
            class_loss = class_loss.item()
            accuracy = model.compute_accuracy(test_data, class_output)
            roc_auc = model.compute_roc_auc(test_data, class_output)

            class_output = class_output.cpu().detach().numpy()

            true_labels = test_data.y.cpu().numpy()

            if class_output.ndim == 1:
                probabilities = class_output
            elif class_output.ndim == 2:
                probabilities = class_output[:,1]
            else:
                raise ValueError("The class outputs have an unexpected shape.")
    
            fpr, tpr, thresholds = metrics.roc_curve(true_labels, probabilities, drop_intermediate=False)

            all_class_outputs.append(class_output)

            all_losses.append(loss)
            all_recon_losses.append(recon_loss)
            all_class_losses.append(class_loss)
            all_accuracies.append(accuracy)
            all_roc_aucs.append(roc_auc)
            all_tprs.append(tpr)
            all_fprs.append(fpr)

            print("Fold finished")

    stacked_class_outputs = np.concatenate(all_class_outputs)

    print(stacked_class_outputs.shape)

    
    all_losses = np.array(all_losses)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)

    all_recon_losses = np.array(all_recon_losses)
    mean_recon_loss = np.mean(all_recon_losses)
    std_recon_loss = np.std(all_recon_losses)

    all_class_losses = np.array(all_class_losses)
    mean_class_loss = np.mean(all_class_losses)
    std_class_loss = np.std(all_class_losses)

    all_accuracies = np.array(all_accuracies)
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    all_roc_aucs = np.array(all_roc_aucs)
    mean_roc_auc = np.mean(all_roc_aucs)
    std_roc_auc = np.std(all_roc_aucs)

    print(tcols.OKCYAN + f"Test loss: {mean_loss:.4f} +/- {std_loss:.4f}" + tcols.ENDC)
    print(tcols.OKCYAN + f"Test reconstruction loss: {mean_recon_loss:.4f} +/- {std_recon_loss:.4f}" + tcols.ENDC)
    print(tcols.OKCYAN + f"Test classification loss: {mean_class_loss:.4f} +/- {std_class_loss:.4f}" + tcols.ENDC)
    print(tcols.OKCYAN + f"Test accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}" + tcols.ENDC)
    print(tcols.OKCYAN + f"Test ROC AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}" + tcols.ENDC)


    return all_roc_aucs, all_class_outputs, all_tprs, all_fprs