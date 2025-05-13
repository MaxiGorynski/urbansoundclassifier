import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class UrbanSoundDataset(Dataset):
    """
    Custom PyTorch Dataset for Urban Sound Classification

    This class handles loading and preprocessing of audio files
    from the UrbanSound8K dataset.
    """

    # Urban Sound 8K known classes
    URBAN_SOUND_CLASSES = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]

    def __init__(self, dataset_dir, transform=None):
        """
        Initialize the dataset

        :param dataset_dir: Directory containing the UrbanSound8K dataset
        :param transform: Optional transform to be applied on a sample
        """
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Audio file paths and corresponding labels
        self.audio_files = []
        self.labels = []

        # Label encoder to convert string labels to integers
        self.label_encoder = LabelEncoder()

        # Find and process audio files
        self._process_dataset()

    def __len__(self):
        """
        Return the number of samples in the dataset

        :return: Total number of audio samples
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Generate one sample of data

        :param idx: Index of the sample
        :return: Mel spectrogram and corresponding label
        """
        # Load audio file
        try:
            waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        except Exception as e:
            print(f"Error loading audio file {self.audio_files[idx]}: {e}")
            # Return a dummy sample to avoid breaking the dataloader
            return torch.zeros(1, 128, 128), torch.tensor(0)

        # Convert to mel spectrogram
        mel_spectrogram = self._audio_to_mel_spectrogram(waveform, sample_rate)

        # Ensure the spectrogram is the right shape and has a channel dimension
        if mel_spectrogram.ndim == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        elif mel_spectrogram.ndim == 3:
            mel_spectrogram = mel_spectrogram.squeeze(0)

        # Ensure shape is exactly 1x128x128
        mel_spectrogram = mel_spectrogram[:1, :128, :128]

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return mel_spectrogram, label

    def _extract_label(self, filename):
        """
        Extract label from filename

        Handles various filename formats:
        1. 101415-3-0-2.wav (numeric prefix)
        2. aircon_001.wav (text prefix)
        """
        # Try extracting from numeric ID
        name_parts = filename.split('-')
        if len(name_parts) > 2:
            try:
                class_id = int(name_parts[1])
                # Ensure class_id is within valid range
                if 0 <= class_id < len(self.URBAN_SOUND_CLASSES):
                    return self.URBAN_SOUND_CLASSES[class_id]
            except (ValueError, IndexError):
                pass

        # Fallback to prefix extraction
        prefix = filename.split('_')[0].lower()

        # Check if prefix matches any known class names
        for known_class in self.URBAN_SOUND_CLASSES:
            if prefix in known_class:
                return known_class

        return 'unknown'

    def _process_dataset(self):
        """
        Process the dataset and collect audio file paths and labels
        Specifically looks in the audio/fold* directories
        """
        # Construct the audio directory path
        audio_dir = os.path.join(self.dataset_dir, 'audio')

        # Check if audio directory exists
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Try to find metadata CSV
        metadata_path = os.path.join(self.dataset_dir, 'urbansound8k.csv')

        # If metadata CSV exists, use it for precise labeling
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)

            for _, row in metadata.iterrows():
                fold = row['fold']
                filename = row['slice_file_name']
                label = row['class']

                # Construct full audio file path
                audio_path = os.path.join(audio_dir, f'fold{fold}', filename)

                if os.path.exists(audio_path):
                    self.audio_files.append(audio_path)
                    self.labels.append(label)

        # Fallback to directory scanning
        else:
            # Scan fold directories
            for fold in range(1, 11):  # 10 folds in UrbanSound8K
                fold_path = os.path.join(audio_dir, f'fold{fold}')

                if not os.path.exists(fold_path):
                    print(f"Warning: Fold directory not found: {fold_path}")
                    continue

                for filename in os.listdir(fold_path):
                    if filename.endswith('.wav'):
                        audio_path = os.path.join(fold_path, filename)
                        self.audio_files.append(audio_path)

                        # Extract label from filename
                        label = self._extract_label(filename)
                        self.labels.append(label)

        # Ensure we found some audio files
        if not self.audio_files:
            raise ValueError("No audio files found. Check dataset directory structure.")

        # Encode labels
        self.labels = self.label_encoder.fit_transform(self.labels)

        print(f"Loaded {len(self.audio_files)} audio files")
        print(f"Unique labels: {self.label_encoder.classes_}")

    def _audio_to_mel_spectrogram(self, waveform, sample_rate):
        """
        Convert audio to mel spectrogram with consistent shape

        :param waveform: Input audio waveform
        :param sample_rate: Sample rate of the audio
        :return: Normalized mel spectrogram
        """
        # Resample if needed
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(sample_rate, 22050)
            waveform = resampler(waveform)
            sample_rate = 22050

        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Trim or pad to 4 seconds
        target_length = 4 * sample_rate
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Create mel spectrogram
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,  # Ensure consistent number of mels
            n_fft=2048,
            hop_length=512
        )

        mel_spectrogram = mel_spectrogram_transform(waveform)

        # Convert to decibels
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # Manually resize to 128x128 using interpolation
        # First, ensure mel_spectrogram_db is a 2D tensor
        mel_spectrogram_db = mel_spectrogram_db.squeeze()

        # Resize using torch.nn.functional interpolate
        mel_spectrogram_resized = torch.nn.functional.interpolate(
            mel_spectrogram_db.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=(128, 128),  # Target size
            mode='bilinear',  # Bilinear interpolation
            align_corners=False
        ).squeeze()

        # Normalize
        mel_spectrogram_normalized = (
                                                 mel_spectrogram_resized - mel_spectrogram_resized.mean()) / mel_spectrogram_resized.std()

        return mel_spectrogram_normalized


class SoundClassificationCNN(nn.Module):
    """
    Convolutional Neural Network for Sound Classification
    """

    def __init__(self, num_classes):
        super(SoundClassificationCNN, self).__init__()

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Determine the feature dimensions dynamically
        def _get_feature_size(input_shape):
            with torch.no_grad():
                test_input = torch.zeros(1, *input_shape)
                features = self.conv_layers(test_input)
                return features.view(1, -1).size(1)

        # Assume input is a mel spectrogram of shape (1, 128, 128)
        feature_size = _get_feature_size((1, 128, 128))

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network

        :param x: Input tensor (mel spectrogram)
        :return: Class predictions
        """
        # Ensure input is the right shape
        if x.ndim == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.shape[1] != 1:
            x = x[:, :1]  # Ensure single channel

        # Extract features
        features = self.conv_layers(x)

        # Flatten features
        x = features.view(features.size(0), -1)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        return x


class UrbanSoundClassifier:
    """
    Main class to handle the entire machine learning pipeline
    """

    def __init__(self, dataset_path, output_path=None, device=None):
        """
        Initialize the Sound Classification model

        :param dataset_path: Path to UrbanSound8K dataset
        :param output_path: Path to save model and results
        :param device: Torch device (CPU/GPU)
        """
        self.dataset_path = dataset_path
        self.output_path = output_path or './sound_classification_output'

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Set up device
        self.device = device if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Create dataset
        self.dataset = UrbanSoundDataset(dataset_path)

        # Split dataset
        self._split_dataset()

        # Initialize model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _split_dataset(self):
        """
        Split dataset into train, validation, and test sets
        """
        total_size = len(self.dataset)
        test_size = int(0.1 * total_size)
        val_size = int(0.2 * (total_size - test_size))
        train_size = total_size - test_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )

        print(f"Dataset split:")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Test samples: {test_size}")

    def create_dataloaders(self, batch_size=32):
        """
        Create DataLoaders for training, validation, and testing

        :param batch_size: Number of samples per batch
        """
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def initialize_model(self):
        """
        Initialize the model, loss function, and optimizer
        """
        # Number of classes from label encoder
        num_classes = len(self.dataset.label_encoder.classes_)

        # Create model
        self.model = SoundClassificationCNN(num_classes).to(self.device)

        # Print model summary
        print("\nModel Architecture:")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total trainable parameters: {total_params:,}")

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )

    def train_epoch(self):
        """
        Train the model for one epoch

        :return: Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        # Progress tracking
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_loss += loss.item()

        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total_samples

        return epoch_loss, epoch_accuracy

    def validate(self):
        """
        Validate the model on the validation dataset

        :return: Validation loss, accuracy, and classification report
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        # Disable gradient computation for validation
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Compute predictions
                _, predicted = torch.max(outputs.data, 1)

                # Track metrics
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

                total_loss += loss.item()

                # Collect predictions for detailed report
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total_samples

        # Generate classification report
        report = sklearn.metrics.classification_report(
            all_labels,
            all_predictions,
            target_names=self.dataset.label_encoder.classes_
        )

        return val_loss, val_accuracy, report

    def train(self, epochs=25):
        """
        Train the entire model

        :param epochs: Number of training epochs
        :return: Dictionary of training metrics
        """
        # Initialize model and dataloaders
        self.initialize_model()
        self.create_dataloaders()

        # Training history tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_accuracy = self.train_epoch()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validate
            val_loss, val_accuracy, val_report = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print epoch results
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_path, 'best_model.pth')
                )
            else:
                patience_counter += 1

            # Stop if no improvement
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Plot training history
        self._plot_training_history(
            train_losses, val_losses,
            train_accuracies, val_accuracies
        )

        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_val_report': val_report
        }

    def _plot_training_history(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """
        Visualize training and validation metrics

        :param train_losses: List of training losses
        :param val_losses: List of validation losses
        :param train_accuracies: List of training accuracies
        :param val_accuracies: List of validation accuracies
        """
        plt.figure(figsize=(12, 5))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'training_history.png'))
        plt.close()

    def evaluate(self):
        """
        Perform final evaluation on the test dataset

        :return: Dictionary of test evaluation metrics
        """
        # Load the best model
        best_model_path = os.path.join(self.output_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # Set model to evaluation mode
        self.model.eval()

        # Tracking variables
        correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        # Disable gradient computation
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)

                # Update metrics
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect predictions for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate test accuracy
        test_accuracy = 100 * correct / total_samples

        # Generate detailed classification report
        classification_report = sklearn.metrics.classification_report(
            all_labels,
            all_predictions,
            target_names=self.dataset.label_encoder.classes_
        )

        # Generate confusion matrix
        confusion_matrix = sklearn.metrics.confusion_matrix(
            all_labels,
            all_predictions
        )

        # Print results
        print("\n--- Test Evaluation Results ---")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report)

        return {
            'test_accuracy': test_accuracy,
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix
        }

    def save_model(self):
        """
        Save the trained model in multiple formats
        """
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Prepare example input for tracing and export
        example_input = torch.randn(1, 1, 128, 128).to(self.device)

        # 1. Save full PyTorch model checkpoint
        full_checkpoint_path = os.path.join(self.output_path, 'full_model_checkpoint.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'label_encoder': self.dataset.label_encoder
        }, full_checkpoint_path)
        print(f"Full model checkpoint saved to: {full_checkpoint_path}")

        # 2. Save TorchScript model for deployment
        traced_model_path = os.path.join(self.output_path, 'traced_model.pt')
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(traced_model_path)
        print(f"TorchScript model saved to: {traced_model_path}")

        # 3. Export to ONNX for cross-platform inference
        onnx_model_path = os.path.join(self.output_path, 'sound_classifier_model.onnx')
        torch.onnx.export(
            self.model,
            example_input,
            onnx_model_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model saved to: {onnx_model_path}")

    def run_pipeline(self):
        """
        Execute the complete machine learning pipeline

        :return: Dictionary of training and evaluation results
        """
        # Train the model
        training_results = self.train()

        # Evaluate the model
        evaluation_results = self.evaluate()

        # Save the model in different formats
        self.save_model()

        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }



def parse_arguments():
    """
    Parse command-line arguments for the Urban Sound Classification project

    :return: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Urban Sound Classification using PyTorch'
    )

    # Required argument for dataset path
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to the UrbanSound8K dataset directory'
    )

    # Optional arguments
    parser.add_argument(
        '--output_path',
        type=str,
        default='./sound_classification_output',
        help='Directory to save model outputs and results'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs (default: 25)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer (default: 0.001)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Explicitly select computation device'
    )

    return parser.parse_args()


def main():
    """
    Main function to run the Urban Sound Classification project
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        sys.exit(1)

    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Start timing the entire process
        start_time = time.time()

        # Initialize the classifier
        classifier = UrbanSoundClassifier(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            device=device
        )

        # Run the full machine learning pipeline
        results = classifier.run_pipeline()

        # Calculate total runtime
        end_time = time.time()
        total_runtime = end_time - start_time

        # Print summary of results
        print("\n--- Project Summary ---")
        print(f"Dataset: {args.dataset_path}")
        print(f"Total Runtime: {total_runtime:.2f} seconds")
        print(f"Best Model Saved to: {os.path.join(args.output_path, 'best_model.pth')}")

        # Print key metrics
        print("\n--- Training Results ---")
        print(f"Final Training Accuracy: {results['training_results']['train_accuracies'][-1]:.2f}%")
        print(f"Final Validation Accuracy: {results['training_results']['val_accuracies'][-1]:.2f}%")

        print("\n--- Test Results ---")
        print(f"Test Accuracy: {results['evaluation_results']['test_accuracy']:.2f}%")

        # Generate a detailed report
        with open(os.path.join(args.output_path, 'classification_report.txt'), 'w') as f:
            f.write("Urban Sound Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write("Training Metrics:\n")
            f.write(f"Total Epochs: {len(results['training_results']['train_accuracies'])}\n")
            f.write(f"Final Training Accuracy: {results['training_results']['train_accuracies'][-1]:.2f}%\n")
            f.write(f"Final Validation Accuracy: {results['training_results']['val_accuracies'][-1]:.2f}%\n\n")
            f.write("Test Results:\n")
            f.write(f"Test Accuracy: {results['evaluation_results']['test_accuracy']:.2f}%\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(results['evaluation_results']['classification_report'])

        print(f"\nDetailed report saved to: {os.path.join(args.output_path, 'classification_report.txt')}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Import time for runtime tracking
    import time

    # Run the main function
    main()