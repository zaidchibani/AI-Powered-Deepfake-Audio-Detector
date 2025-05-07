import numpy as np
import librosa
from pydub import AudioSegment
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def load_saved_model(model_path):
    """
    Load a saved Keras model from disk.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_single_audio(audio_path):
    """
    Preprocess a single audio file the same way as in training.
    
    Parameters:
    - audio_path: Path to the audio file
    
    Returns:
    - Feature vector ready for model input
    """
    try:
        # Load and preprocess audio
        raw_audio = AudioSegment.from_file(audio_path)
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
        trimmed, _ = librosa.effects.trim(samples, top_db=25)
        padding = max(0, 50000 - len(trimmed))
        padded = np.pad(trimmed, (0, padding), 'constant')
        
        # Extract features
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        
        # Extract ZCR
        zcr = librosa.feature.zero_crossing_rate(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        zcr = np.expand_dims(zcr, axis=0)
        
        # Extract RMS
        rms = librosa.feature.rms(y=padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = np.expand_dims(rms, axis=0)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=padded, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)
        mfccs = np.expand_dims(mfccs, axis=0)
        
        # Combine features
        zcr_features = np.swapaxes(zcr, 1, 2)
        rms_features = np.swapaxes(rms, 1, 2)
        mfccs_features = np.swapaxes(mfccs, 1, 2)
        
        X_features = np.concatenate((zcr_features, rms_features, mfccs_features), axis=2)
        
        return X_features
        
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

def test_single_audio(model, audio_path):
    """
    Test a single audio file with the model.
    
    Parameters:
    - model: The loaded model object
    - audio_path: Path to the audio file
    
    Returns:
    - predicted_class: 'real' or 'fake'
    - confidence: Confidence score
    """
    X_features = preprocess_single_audio(audio_path)
    
    if X_features is None:
        return None, None
    
    # Make prediction
    prediction = model.predict(X_features, verbose=0)[0]
    
    # Get class and confidence
    predicted_class_idx = np.argmax(prediction)
    confidence = prediction[predicted_class_idx]
    
    # Map class index to label
    predicted_class = 'real' if predicted_class_idx >= 0.8 else 'fake'
    
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_class, confidence

def test_single_audio(model, audio_path):
    """
    Predict if a single audio file is real or fake.
    """
    features = preprocess_single_audio(audio_path)
    if features is None:
        return None, None

    prediction = model.predict(features, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    predicted_class = 'real' if predicted_index == 1 else 'fake'

    print(f"{audio_path} → {predicted_class} ({confidence:.4f})")
    return predicted_class, confidence

def batch_test(model, directory_path):
    """
    Test a directory of audios and print evaluation metrics.
    """
    results = []
    y_true, y_pred = [], []

    has_labels = os.path.isdir(os.path.join(directory_path, 'real')) and os.path.isdir(os.path.join(directory_path, 'fake'))

    categories = ['real', 'fake'] if has_labels else ['']

    for label in categories:
        dir_path = os.path.join(directory_path, label) if label else directory_path
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                path = os.path.join(dir_path, fname)
                prediction, confidence = test_single_audio(model, path)
                if prediction:
                    results.append({
                        'file': fname,
                        'prediction': prediction,
                        'confidence': confidence,
                        'true_label': label if has_labels else None,
                        'correct': (label == prediction) if has_labels else None
                    })
                    if has_labels:
                        y_true.append(1 if label == 'real' else 0)
                        y_pred.append(1 if prediction == 'real' else 0)

    print(f"\nAnalyzed: {len(results)} files")
    if has_labels:
        print(f"Accuracy: {sum(r['correct'] for r in results) / len(results):.2f}")
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['fake', 'real']))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Fake', 'Real'])
        plt.yticks([0, 1], ['Fake', 'Real'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    return results
