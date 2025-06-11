import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def load_values(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file]
    return np.array(values)

models = {
    "Uncompressed GNN": ("trained_classifiers/uncompressed_fixed_full_bce/ClassicalGNN_lr0.1_batch32", "blue"),
    "Not Guided SAG model + GNN": ("trained_classifiers/notguided_SAG_model_fixed_full_classical_cpu/SAG_model_ClassicalGNN_lr0.1_batch32", "green"),
    "Not Guided SAG model + QGNN2": ("trained_classifiers/notguided_fixed_full_cpu_paper/SAG_model_QGNN2_lr0.1_batch32_layers6", "red"),
    "Guided SAG model + GNN": ("trained_guided_classifiers/guided_SAG_model_classical_cpu_bce/SAG_model_ClassicalGNN_lr0.01_batch256_class_weight_0.8", "purple"),
    "Guided SAG model + QGNN2": ("trained_guided_classifiers/guided_SAG_model_QGNN2/SAG_model_QGNN2_lr0.001_batch32_layers6_class_weight_0.8", "orange"),
}

plt.figure(figsize=(12,10))

for model_name, (path, color) in models.items():

    npz_path = os.path.join(path, "roc_plots/test_results_data.npz")
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        continue

    data = np.load(npz_path)
    mean_fpr = data["mean_fpr"]
    mean_tpr = data["mean_tpr"]
    std_tpr = data["std_tpr"]
    mean_roc_auc = data["mean_roc_auc"]
    std_roc_auc = data["std_roc_auc"]

    # Plot the mean ROC curve for this model.
    plt.plot(mean_fpr, mean_tpr, color=color,
             label=f"{model_name} (AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f})")
    
    # Plot the standard deviation as a shaded region.
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color=color, alpha=0.2)

# Plot the diagonal line for random chance
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Add plot details
plt.xlabel('False Positive Rate', fontsize=17)
plt.ylabel('True Positive Rate', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid(True)
plt.tight_layout()
plt.savefig('SAG_model_roc_curves.svg', bbox_inches='tight')

# Display the plot
#plt.show()