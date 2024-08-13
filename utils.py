import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from classes import Encoder


def save_encoder(model: nn.Module, save_path) -> None:
    encoder = model.encoder.state_dict()
    torch.save(encoder, save_path)



def load_encoder(load_path):
    encoder = Encoder()
    state_dict = torch.load(load_path)
    encoder.load_state_dict(state_dict)
    return encoder



# Function to get data loaders for training, validation, and testing
def get_dataloaders(batch_size=64):
    # Define a transform to convert data to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the MNIST training and testing datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_small_train_loader(n=100):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = datasets.MNIST("data",
                                  train=True,
                                  download=True,
                                  transform=transform)
    indices = torch.arange(n)
    train_loader_CLS = Subset(train_loader, indices)

    train_loader_CLS = torch.utils.data.DataLoader(train_loader_CLS, batch_size=64, shuffle=True, num_workers=0)
    return train_loader_CLS


# Function to split the training dataset into training and validation sets
def split_train_val(dataloader, val_split=0.3, batch_size=64):
    # Extract the dataset from the DataLoader
    dataset = dataloader.dataset

    # Use sklearn train_test_split to split the indices
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=val_split)

    # Create Subset objects
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoaders for the subsets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_ae_outputs(autoencoder, test_loader, save_path, examples_per_digit=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder.to(device)
    autoencoder.eval()

    digit_indices = find_digit_indices(test_loader, examples_per_digit)
    test_dataset = test_loader.dataset

    fig, axes = plt.subplots(2 * examples_per_digit, 10, figsize=(15, 8))

    for digit in range(10):
        for j in range(examples_per_digit):
            idx = digit_indices[digit][j]
            img, _ = test_dataset[idx]
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                rec_img = autoencoder(img)

            # Plot original image
            ax = axes[2 * j, digit]
            ax.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Digit {digit}')

            # Plot reconstructed image
            ax = axes[2 * j + 1, digit]
            ax.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.axis('off')
            if j == 0:
                ax.set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def find_digit_indices(data_loader, examples_per_digit=3):
    digit_indices = {i: [] for i in range(10)}
    for batch_idx, (data, targets) in enumerate(data_loader):
        for i, target in enumerate(targets):
            digit = target.item()
            if len(digit_indices[digit]) < examples_per_digit:
                digit_indices[digit].append(batch_idx * data_loader.batch_size + i)
            if all(len(indices) == examples_per_digit for indices in digit_indices.values()):
                break
        if all(len(indices) == examples_per_digit for indices in digit_indices.values()):
            break
    return digit_indices

# Function to train the model in an unsupervised manner
def unsupervised_train(model, dataloader, epochs, lr=1e-3, save_path=None) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()  # Mean L1 loss for reconstruction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    last_epoch_loss = 0.0
    for epoch in range(epochs):
        loss_sum = 0.0
        for data in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        last_epoch_loss = loss_sum / len(dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {last_epoch_loss:.4f}')
    if save_path:
        save_encoder(model, save_path)
    return last_epoch_loss


# Function to evaluate the model in an unsupervised manner
def unsupervised_evaluate(autoencoder, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()  # Mean L1 loss for reconstruction
    total_loss = 0
    autoencoder.eval()
    autoencoder.to(device)
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            output = autoencoder(img)
            loss = criterion(output, img)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(dataloader):.4f}')
    return total_loss / len(dataloader)


# Function to train the classifier in a supervised manner
def supervised_train(model, train_loader, criterion, optimizer, test_loader=None, epochs=10, should_evaluate=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    train_losses = []
    train_acc = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss /= len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        if should_evaluate:
            test_loss, test_acc = supervised_evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    return train_losses, train_acc, test_losses, test_accuracies


# Function to evaluate the classifier in a supervised manner
def supervised_evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return test_loss, test_acc


# Function to plot training and testing metrics
def plot_metrics(train_loss, train_acc, test_loss=None, test_acc=None, with_eval=False, file_name=None):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    if with_eval:
        plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    if with_eval:
        plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.savefig(file_name)
    plt.close()


def train_and_plot_unsupervised(model, epochs, save_path):
    train_loader, test_loader = get_dataloaders()
    unsupervised_train(model, train_loader, epochs=epochs, lr=1e-3)
    unsupervised_evaluate(model, test_loader)
    plot_ae_outputs(model, test_loader, save_path)
