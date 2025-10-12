import os
import urllib.request
import bz2
import shutil

models_dir = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\models\Dlib"
os.makedirs(models_dir, exist_ok=True)

models = {
    "mmod_human_face_detector.dat": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
    "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

for filename, url in models.items():
    bz2_path = os.path.join(models_dir, filename + ".bz2")
    dat_path = os.path.join(models_dir, filename)
    
    if os.path.exists(dat_path):
        print(f"✓ {filename} already exists")
        continue
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, bz2_path)
    
    print(f"Extracting {filename}...")
    with bz2.BZ2File(bz2_path, 'rb') as f_in:
        with open(dat_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(bz2_path)
    print(f"✓ {filename} ready")

print("All Dlib models downloaded and extracted!")