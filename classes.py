import torch.nn as nn
import torch


# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim=12):
        super().__init__()
        # Convolutional layers followed by a fully connected layer
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # Input channels: 1, Output channels: 8
            nn.ReLU(True),  # Activation function
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),  # Input channels: 16, Output channels: 32
            nn.ReLU(True),
            nn.Flatten(start_dim=1),  # Flatten the tensor
            nn.Linear(3 * 3 * 32, encoded_space_dim)  # Fully connected layer
        )

    def forward(self, x):
        return self.encoder_cnn(x)  # Forward pass through the encoder


# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, encoded_space_dim=12):
        super().__init__()
        # Fully connected layer followed by transposed convolutional layers
        self.decoder_conv = nn.Sequential(
            nn.Linear(encoded_space_dim, 3 * 3 * 32),  # Fully connected layer
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),  # Unflatten the tensor
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),  # Transposed convolution
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # Transposed convolution
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)  # Final transposed convolution
        )

    def forward(self, x):
        return torch.sigmoid(self.decoder_conv(x))  # Apply sigmoid activation to output


# Define the MLP (Multi-Layer Perceptron) class
class MLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=None, output_dim=12):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))  # Fully connected layer
            layers.append(nn.ReLU(True))  # Activation function
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))  # Final fully connected layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


# Define the ConvAutoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, mlp_hidden_dims):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder()  # Encoder part
        self.mlp = MLP(hidden_dims=mlp_hidden_dims)  # MLP part
        self.decoder = Decoder()  # Decoder part

    def forward(self, x):
        x = self.encoder(x)  # Forward pass through encoder
        x = self.mlp(x)  # Forward pass through MLP
        x = self.decoder(x)  # Forward pass through decoder
        return x


# Define the Classifier class
class Classifier(nn.Module):
    def __init__(self, hidden_dim, pre_trained_encoder=None):
        super(Classifier, self).__init__()
        self.encoder = Encoder() if pre_trained_encoder is None else pre_trained_encoder  # Encoder part
        self.mlp = MLP(hidden_dims=[hidden_dim], output_dim=10)  # MLP part for classification

    def forward(self, x):
        return self.mlp(self.encoder(x))  # Forward pass through encoder and MLP


# Define the ClassifierDecoder class
class ClassifierDecoder(nn.Module):
    def __init__(self, encoder):
        super(ClassifierDecoder, self).__init__()
        self.encoder = encoder  # Pre-trained encoder part
        self.decoder = Decoder()  # Decoder part
        self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.required_grad = False  # Freeze encoder parameters

    def forward(self, x):
        return self.decoder(self.encoder(x))  # Forward pass through encoder and decoder
