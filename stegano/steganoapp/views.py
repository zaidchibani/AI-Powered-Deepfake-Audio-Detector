# views.py
import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from .steganomodel import load_saved_model, test_single_audio  # Adjust import as needed

# Load your models only once
MODEL = load_saved_model(r'C:\Users\zaidc\Desktop\cyber\stegano\steganoapp\deepfakedetectioncyber.keras')
MODEL1 = load_saved_model(r'C:\Users\zaidc\Desktop\cyber\stegano\steganoapp\deepfakedetection.keras')

def detect_audio(request):
    result = None
    error = None
    
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        file_path = default_storage.save('temp_audio/' + audio_file.name, audio_file)
        abs_path = default_storage.path(file_path)
        
        try:
            # Check which model to use based on form input
            model_type = request.POST.get('model_type', 'standard')
            
            if model_type == 'advanced':
                # Use the advanced model (MODEL)
                model_to_use = MODEL
                model_name = "Advanced AI Detection"
            else:
                # Use the standard model (MODEL1)
                model_to_use = MODEL1
                model_name = "Standard Detection"
            
            predicted_class, confidence = test_single_audio(model_to_use, abs_path)
            
            if predicted_class is not None:
                result = f"[{model_name}] Prediction: {predicted_class.upper()} | Confidence: {confidence*100:.2f}%"
            else:
                error = "Error analyzing the audio. Please try again."
        except Exception as e:
            error = str(e)
        finally:
            # Cleanup
            if os.path.exists(abs_path):
                os.remove(abs_path)
    
    return render(request, 'index.html', {
        'result': result,
        'error': error
    })