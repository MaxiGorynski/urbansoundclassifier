import os
import sys
import logging
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log system information
logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"NumPy Version: {np.__version__}")
logger.info(f"Torchaudio Version: {torchaudio.__version__}")


# Recreate the CNN architecture (must match original model)
class SoundClassificationCNN(nn.Module):
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


class SoundClassifier:
    def __init__(self, model_path):
        """
        Initialize sound classifier

        :param model_path: Path to the trained model checkpoint
        """
        # Known classes from original training
        self.classes = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]

        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model architecture
        self.model = SoundClassificationCNN(len(self.classes))

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # If the state dict has a 'model_state_dict' key (from full checkpoint)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _audio_to_mel_spectrogram(self, waveform, sample_rate):
        """
        Convert audio to mel spectrogram

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
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

        mel_spectrogram = mel_spectrogram_transform(waveform)

        # Convert to decibels
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # Manually resize to 128x128 using interpolation
        mel_spectrogram_db = mel_spectrogram_db.squeeze()

        # Resize using torch.nn.functional interpolate
        mel_spectrogram_resized = torch.nn.functional.interpolate(
            mel_spectrogram_db.unsqueeze(0).unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Normalize
        mel_spectrogram_normalized = (
                                                 mel_spectrogram_resized - mel_spectrogram_resized.mean()) / mel_spectrogram_resized.std()

        return mel_spectrogram_normalized

    def predict(self, audio_path):
        """
        Predict sound class for an audio file

        :param audio_path: Path to audio file
        :return: Prediction results
        """
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mel spectrogram
            mel_spectrogram = self._audio_to_mel_spectrogram(waveform, sample_rate)

            # Add batch and channel dimensions
            mel_spectrogram = mel_spectrogram.unsqueeze(0).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = self.model(mel_spectrogram)
                probabilities = torch.softmax(outputs, dim=1)[0]

                # Get top 3 predictions
                top_3_prob, top_3_indices = torch.topk(probabilities, 3)

                predictions = [
                    {
                        'class': self.classes[idx],
                        'probability': prob.item()
                    }
                    for idx, prob in zip(top_3_indices, top_3_prob)
                ]

                return predictions

        except Exception as e:
            return {'error': str(e)}


# Flask API
app = Flask(__name__)

# Initialize model (adjust path as needed)
MODEL_PATH = './sound_classification_output/best_model.pth'


@app.route('/', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Sound Classification Endpoint is running',
        'model_path': MODEL_PATH
    }), 200


@app.route('/predict', methods=['POST'])
def predict_sound():
    """
    Endpoint for sound classification

    Expects audio file in request
    """
    try:
        # Initialize classifier here to catch any initialization errors
        classifier = SoundClassifier(MODEL_PATH)
    except Exception as init_error:
        logger.error(f"Model initialization error: {init_error}")
        return jsonify({
            'error': 'Failed to initialize model',
            'details': str(init_error)
        }), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        file.save(temp_file.name)

        try:
            # Predict
            predictions = classifier.predict(temp_file.name)

            # Clean up temporary file
            os.unlink(temp_file.name)

            return jsonify(predictions)

        except Exception as e:
            # Ensure temp file is deleted even if prediction fails
            os.unlink(temp_file.name)
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Sound Classification Endpoint')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the server')
    parser.add_argument('--port', type=int, default=5005,
                        help='Port to run the server')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Log startup information
    logger.info(f"Starting server on {args.host}:{args.port}")

    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=True)