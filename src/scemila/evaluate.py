import pandas as pd
import torch
import os
import json
from pathlib import Path
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.transforms import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from src.utils import get_results_path, check_gpu
from src.scemila.model import ImageClassificationModel


def plot_confusion_matrix(predictions, y_test_, save_dir):
    """ Build the confusion matrix from the model and grounds truth

    :param model_:
    :param x_test_:
    :param y_test_:
    :return:
    """

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.8)  # Adjust font scale for better readability

    cm = confusion_matrix(y_test_, predictions)
    labels = np.unique(y_test_)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ground_truth_counts = {label: np.sum(y_test == label) for label in np.unique(y_test_)}

    # Get non-zero elements and their indices
    nonzero_indices = cm_norm > 0.001
    nonzero_values = cm_norm[nonzero_indices]
    non_zero_labels = [labels[i] for i in nonzero_indices[0]]

    # Plot non-zero values
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    # Add annotations for number of points in ground truth for each class
    for i, label in enumerate(labels):
        plt.text(-2.5, i + 0.5, f'n={ground_truth_counts[label]}',
                 ha='center', va='center', color='red')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()


def plot_loss(data_dir_, save_dir):
    # Read the text file
    with open(data_dir_ / "train.log", 'r') as file:
        lines = file.readlines()

    # Extract values from each line
    data = []
    for line in lines:
        line = line.strip().split('::')[1]  # Remove "loss::" and split by commas
        values = line.split(',')  # Split values
        data.append(values)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['epoch', 'val_loss', 'train_loss'])

    # Convert columns to numeric if needed
    df['epoch'] = pd.to_numeric(df['epoch'])
    df['val_loss'] = pd.to_numeric(df['val_loss'])
    df['train_loss'] = pd.to_numeric(df['train_loss'])

    plt.figure()
    plt.plot(df.epoch, df.val_loss, label="val loss")
    plt.plot(df.epoch, df.train_loss, label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(results_dir / "loss.png")


def plot_clustering(activations_, ground_truth_, save_dir_):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    projections = pca.fit_transform(activations_)

    # Create a colormap for the number of labels
    unique_labels = np.unique(ground_truth_)
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3'
    ]
    plt.figure(figsize=(10, 7))

    for i, label in enumerate(unique_labels):
        idx = ground_truth_ == label
        plt.scatter(projections[idx, 0], projections[idx, 1], color=colors[i], label=label, s=1)

    plt.title('PCA Projected Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir_ / "clustering.png")


def explainability(model_, x_test_, highest_prob, preprocess, save_dir_):
    """

    :return:
    """
    def forward_model(input, model_):
        return torch.nn.Softmax(dim=1)(model_(input))

    integrated_gradients = IntegratedGradients(forward_model)
    for label, high_prob in highest_prob.items():
        prob, idx = high_prob
        label = int(label)
        x_input = preprocess(x_test_[idx]).unsqueeze(0)
        attributions_ig = integrated_gradients.attribute(x_input, target=label, n_steps=600, additional_forward_args=model_)  # think about target or baselines

        # Convert attributions to numpy for visualization
        attributions_ig_np = attributions_ig.squeeze().cpu().detach().numpy()
        attributions_ig_np = attributions_ig_np.transpose(1, 2, 0)

        # Visualize
        fig = viz.visualize_image_attr_multiple(attributions_ig_np,
                                          x_test_[idx],
                                          methods=["original_image", "heat_map"],
                                          signs=["all", "positive"],
                                          cmap='inferno',
                                          show_colorbar=True,
                                          use_pyplot=False)

        ig_dir = save_dir_ / "integrated_gradients"
        os.makedirs(ig_dir)
        fig.savefig(ig_dir / f"ig_{label}.png")

    raise NotImplementedError()


def build_dir(training):
    dir_ = Path("nucleus_classification") / training
    os.makedirs(dir_, exist_ok=True)
    return dir_


if __name__ == "__main__":

    # files that are assumed to be inside the training directory:
    # training.log - plot the visualization curves
    # training_params.json - the params that are needed to create the model
    # model_params.pth - the params that needs to be load from inside the model
    # parameters.pth - the parameters of the pretrained model to load inside the model

    training_name = "model_0.0001_100_224_vit_32+stardist_qupath_patch-HE_iou-0.5-1.0_pca-100_neigh_30_train.pkl"
    dataset_name = "stardist_qupath_patch-HE_iou-0.5-1.0_pca-100_neigh_30_test.pkl"
    sample = False

    # ---------------------------------------------------------------------- #
    # Load Predefined Information

    # General setup
    training_dir = get_results_path() / "scemila" / "training" / training_name
    dataset_dir = get_results_path() / "scemila" / "datasets" / dataset_name

    results_dir = build_dir(training_name)
    device = check_gpu()

    # Extract training and model params
    with open(training_dir / "training_params.json", "r") as file:
        training_params = json.load(file)
    with open(training_dir / "model_params.json", "r") as file:
        model_params = json.load(file)

    # Build the model based on the params
    model = ImageClassificationModel(**model_params)
    model.load(training_dir)
    model.eval()

    # Load test dataset for evaluation
    with open(dataset_dir, "rb") as file:
        x_test, y_test = pickle.load(file)
    y_test = np.array(y_test)

    if sample:
        x_test = x_test[0:1000]
        y_test = y_test[0:1000]

    # Build preprocess with same value as pretrained params
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((training_params["size"], training_params["size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---------------------------------------------------------------------- #
    # Evaluate the models in different metrics

    plot_loss(training_dir, results_dir)

    global activation

    def hook(model, input, output):
        global activation
        activation = output.detach()
    model.model.backbone.encoder.register_forward_hook(hook)

    predictions = []
    activations = []
    highest_prob = {i: [0., 0] for i in np.unique(y_test)}
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        logits = torch.nn.Softmax(dim=1)(model(preprocess(x).unsqueeze(0)).detach())
        if logits.argmax(dim=1) == y:
            if highest_prob[y][0] < logits[0, y]:
                highest_prob[y][0] = float(logits[0, y])
                highest_prob[y][1] = i
        predictions.append(logits.argmax(dim=1).numpy())
        activations.append(activation.view(-1).numpy())

    predictions = np.array(predictions)
    activations = np.array(activations)

    plot_clustering(activations, y_test, results_dir)

    plot_confusion_matrix(predictions, y_test, results_dir)

    explainability(model, x_test, highest_prob, preprocess, results_dir)
