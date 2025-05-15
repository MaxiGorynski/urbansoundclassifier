import os
import torch
import torchaudio
import logging
from flask import Flask, request, jsonify
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model and sound classification
class SoundClassificationCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SoundClassificationCNN, self).__init__()

        # Convolutional Feature Extractor with Adaptive Parameters
        self.feature_extractor = torch.nn.Sequential(
            # First Convolutional Block
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.2),  # Spatial Dropout

            # Second Convolutional Block
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.3),  # Increased Dropout

            # Third Convolutional Block with Reduced Complexity
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.4)  # Higher Dropout
        )

        # Dynamically calculate feature size
        def _get_feature_size(input_shape):
            with torch.no_grad():
                test_input = torch.zeros(1, *input_shape)
                features = self.feature_extractor(test_input)
                return features.view(1, -1).size(1)

        # Calculate feature size for input (1, 128, 128)
        feature_size = _get_feature_size((1, 128, 128))

        # Classifier with strong regularization
        self.classifier = torch.nn.Sequential(
            # First dense layer with L2 regularization
            torch.nn.Linear(feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            # Second dense layer
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.4),

            # Final classification layer
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Ensure single channel input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Feature extraction with dropout
        features = self.feature_extractor(x)

        # Flatten features
        features = features.view(features.size(0), -1)

        # Classification with dropout
        return self.classifier(features)


class SoundClassifier:
    def __init__(self, model_path):
        """
        Initialize sound classifier with detailed logging
        """
        # Known classes
        self.classes = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
            'siren', 'street_music'
        ]

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        try:
            # Check model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Detailed checkpoint investigation
            logger.info("Checkpoint Keys:")
            for key in checkpoint.keys():
                logger.info(f"  {key}")

            # Determine the correct state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Using 'model_state_dict'")
            else:
                state_dict = checkpoint
                logger.info("Using entire checkpoint as state dict")

            # Log state dict keys
            logger.info("State Dict Keys:")
            for key, tensor in state_dict.items():
                logger.info(f"  {key}: {tensor.shape}")

            # Create model with the exact architecture
            self.model = SoundClassificationCNN(len(self.classes)).to(self.device)

            # Try loading with different strategies
            try:
                # First, try strict loading
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                # If strict fails, try partial loading
                logger.warning("Strict loading failed. Attempting partial loading.")
                model_dict = self.model.state_dict()

                # Filter out incompatible keys
                filtered_state_dict = {k: v for k, v in state_dict.items()
                                       if k in model_dict and v.shape == model_dict[k].shape}

                # Update model dict
                model_dict.update(filtered_state_dict)

                # Load with some flexibility
                self.model.load_state_dict(model_dict, strict=False)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _preprocess_audio(self, audio_path):
        """
        Convert audio to mel spectrogram with more robust preprocessing
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(sample_rate, 22050)
            waveform = resampler(waveform)
            sample_rate = 22050

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_spec = mel_transform(waveform)

        # Convert to decibels and resize
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec_resized = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Normalize
        mel_spec_normalized = (mel_spec_resized - mel_spec_resized.mean()) / mel_spec_resized.std()

        return mel_spec_normalized

    def predict(self, audio_path):
        """
        Predict sound class for an audio file with detailed logging
        """
        try:
            # Preprocess audio
            mel_spec = self._preprocess_audio(audio_path)

            # Ensure correct shape: [batch_size, channels, height, width]
            input_tensor = mel_spec.unsqueeze(0).to(self.device)

            # Log input tensor details
            logger.info(f"Input Tensor Shape: {input_tensor.shape}")

            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logger.info(f"Raw Outputs: {outputs}")

                probabilities = torch.softmax(outputs, dim=1)[0]
                logger.info(f"Probabilities: {probabilities}")

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
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


# Flask Application
app = Flask(__name__)

# Hardcoded model path
MODEL_PATH = '/Users/supriyarai/Code/urbansoundclassifier/sound_classification_output/best_model.pth'

# Initialize classifier globally
try:
    classifier = SoundClassifier(MODEL_PATH)
    logger.info("Classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {e}")
    classifier = None


@app.route('/predict', methods=['POST'])
def predict_sound():
    # Check if classifier is loaded
    if not classifier:
        return jsonify({
            'error': 'Classifier not initialized',
            'details': 'Model could not be loaded'
        }), 500

    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        request.files['file'].save(temp_file.name)

    try:
        # Predict
        predictions = classifier.predict(temp_file.name)

        # Clean up temporary file
        os.unlink(temp_file.name)

        return jsonify(predictions)

    except Exception as e:
        # Ensure temp file is deleted
        os.unlink(temp_file.name)
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005, debug=True)