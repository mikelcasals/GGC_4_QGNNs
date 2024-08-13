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

    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs)//args["num_kfolds"], shuffle=False)

    #Autoencoder model definition
    model = util.choose_guided_classifier_model(hp["gae_type"], hp["classifier_type"], device, hp)

    model.load_model(args["model_path"])
    
    start_time = time.time()

    output_folder = "roc_plots/"
    plot_roc_curve(test_graphs, args["num_kfolds"], model, args["model_path"], output_folder)
          
    end_time = time.time()

    train_time = (end_time - start_time) / 60 

    print(tcols.OKCYAN + f"Testing time: {train_time:.2e} mins." + tcols.ENDC)

def plot_roc_curve(test_graphs, num_kfolds, model, model_path, output_folder):

    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs)//num_kfolds, shuffle=False)

    mean_loss, std_loss, mean_recon_loss, std_recon_loss, mean_class_loss, std_class_loss, mean_acc, std_acc, mean_roc_auc, std_roc_auc, class_outputs = test_kfold_classifier(model,test_loader, num_kfolds)

    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)

    test_data = next(iter(test_loader))
    true_labels = test_data.y.cpu().numpy()

    if class_outputs.ndim == 1:
        probabilities = class_outputs
    elif class_outputs.ndim == 2:
        probabilities = class_outputs[:,1]
    else:
        raise ValueError("The class outputs have an unexpected shape.")
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, probabilities)

    

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    fig = plt.figure(figsize=(12, 10))
    
    plt.plot(fpr, tpr, label=f"AUC: {mean_roc_auc:.3f} ± {std_roc_auc:.3f}", color="navy")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()

    fig.savefig(plots_folder + f"roc_plot.pdf")
    np.savetxt(plots_folder + f"fpr_values.txt", fpr, fmt="%f")
    np.savetxt(plots_folder + f"tpr_values.txt", tpr, fmt="%f")

    plt.close()



def test_kfold_classifier(model,test_loader, num_folds):

    all_losses = []
    all_recon_losses = []
    all_class_losses = []
    all_accuracies = []
    all_roc_aucs = []
    all_class_outputs = []
    for test_data in test_loader:
        loss, recon_loss, class_loss, class_output = model.compute_loss(test_data)
        loss = loss.item()
        recon_loss = recon_loss.item()
        class_loss = class_loss.item()
        accuracy = model.compute_accuracy(test_data, class_output)
        roc_auc = model.compute_roc_auc(test_data, class_output)

        class_output = class_output.cpu().detach().numpy()

        all_class_outputs.append(class_output)

        all_losses.append(loss)
        all_recon_losses.append(recon_loss)
        all_class_losses.append(class_loss)
        all_accuracies.append(accuracy)
        all_roc_aucs.append(roc_auc)

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


    return mean_loss, std_loss, mean_recon_loss, std_recon_loss, mean_class_loss, std_class_loss, mean_accuracy, std_accuracy, mean_roc_auc, std_roc_auc, stacked_class_outputs