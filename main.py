import torch.nn as nn
from torch import optim
from classes import ConvAutoencoder, Classifier, ClassifierDecoder
from utils import get_dataloaders, unsupervised_train, unsupervised_evaluate, supervised_train, \
    plot_metrics, train_and_plot_unsupervised, get_small_train_loader, split_train_val, save_encoder, load_encoder


def autoencoder_question(try_epochs=1, final_epochs=20):
    # Define various MLP architectures to test
    mlp_architectures = [
        [128, 64, 32],
        [256, 128, 64],
        [512, 256, 128],
        [128, 64, 16],
        [256, 128, 32],
        [512, 256, 64],
        [128, 32, 16],
        [256, 64, 32]
    ]

    # Get training data loader
    train_loader, _ = get_dataloaders()
    best_loss = float('inf')
    best_architecture = None

    # Split data into training and validation sets (70/30 split)
    sub_train_loader, val_loader = split_train_val(train_loader)

    # Train and validate autoencoders with different MLP architectures
    for i, mlp_hidden_dims in enumerate(mlp_architectures):
        print(f'\nTesting MLP Architecture {i + 1}: {mlp_hidden_dims}')
        autoencoder = ConvAutoencoder(mlp_hidden_dims)
        unsupervised_train(autoencoder, sub_train_loader, epochs=try_epochs, lr=1e-3)
        val_loss = unsupervised_evaluate(autoencoder, val_loader)

        # Keep track of the best performing architecture
        if val_loss < best_loss:
            best_loss = val_loss
            best_architecture = mlp_hidden_dims

    print(f'\nBest MLP Architecture: {best_architecture} with Loss: {best_loss:.4f}')

    # Train the best model on the entire training set and test it
    best_autoencoder = ConvAutoencoder(best_architecture)
    save_path_plot = "q1_plot.png"
    train_and_plot_unsupervised(best_autoencoder, epochs=final_epochs, save_path=save_path_plot)
    save_path_encoder = "encoder_q1.pth"
    save_encoder(best_autoencoder, save_path_encoder)


def classifier_question(hidden_dim=128, epochs=1):
    # Initialize the classifier model
    model = Classifier(hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Get training and test data loaders
    train_loader, test_loader = get_dataloaders()

    # Train the classifier and plot training and test metrics
    train_loss, train_acc, test_loss, test_acc = supervised_train(model, train_loader, criterion,
                                                                  optimizer, test_loader=test_loader,
                                                                  epochs=epochs)
    plot_metrics(train_loss, train_acc, test_loss, test_acc, file_name="q2_plot.png")

    # Save the trained encoder part of the classifier
    save_path = "encoder_q2.pth"
    save_encoder(model, save_path)


def classifier_decoding_question(epochs=20):
    # Load the pre-trained encoder from the classifier
    load_path = "encoder_q2.pth"
    pretrained_encoder = load_encoder(load_path)

    # Initialize the decoder model using the pre-trained encoder
    classifier_decoder = ClassifierDecoder(pretrained_encoder)
    save_path = "q3_plot.png"

    # Train the decoder to minimize reconstruction loss and plot the results
    train_and_plot_unsupervised(classifier_decoder, epochs=epochs, save_path=save_path)


def shortage_in_training_example(hidden_dim=128, epochs=30):
    # Initialize the classifier model
    model = Classifier(hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Get a small subset of the training data (100 examples)
    train_loader = get_small_train_loader(n=100)
    _, test_loader = get_dataloaders()
    # Train the classifier on the small dataset and plot the metrics
    train_loss, train_acc, test_loss, test_acc = supervised_train(model, train_loader, criterion,
                                                                  optimizer, epochs=epochs, test_loader=test_loader)
    plot_metrics(train_loss, train_acc, test_loss, test_acc, with_eval=True, file_name="q4_plot.png")


def transfer_learning_via_fine_tuning(hidden_dim=128, epochs=30):
    # Load the pre-trained encoder from the autoencoder
    load_path = "encoder_q1.pth"
    pretrained_encoder = load_encoder(load_path)

    # Initialize the classifier model with the pre-trained encoder
    model = Classifier(hidden_dim=hidden_dim, pre_trained_encoder=pretrained_encoder)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Get a small subset of the training data (100 examples)
    train_loader = get_small_train_loader(n=100)
    _, test_loader = get_dataloaders()
    # Train the classifier on the small dataset and plot the metrics
    train_loss, train_acc, test_loss, test_acc = supervised_train(model, train_loader, criterion,
                                                                  optimizer, epochs=epochs, test_loader=test_loader)
    plot_metrics(train_loss, train_acc, test_loss, test_acc, with_eval=True, file_name="q5_plot.png")


if __name__ == '__main__':
    # Run the autoencoder training and testing
    autoencoder_question(try_epochs=5, final_epochs=20)

    # Run the classifier training and testing
    classifier_question(epochs=20)

    # Run the classifier decoding and reconstruction loss minimization
    classifier_decoding_question(epochs=20)

    # Run the classifier training with a small training set and observe over-fitting
    shortage_in_training_example(epochs=30)

    # Run transfer learning by fine-tuning the pre-trained encoder on a small dataset
    transfer_learning_via_fine_tuning(epochs=30)
