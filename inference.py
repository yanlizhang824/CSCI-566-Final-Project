import time
start_time = time.time()
from pathlib import Path
import random
import os
import sys
import glob
import librosa
import librosa.display
import simplejpeg
import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio, display
# sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))
sys.path.append(os.path.join(os.getcwd(), 'AudioCLIP-master'))
from model import AudioCLIP
from utils.transforms import ToTensor1D
from torchvision import datasets
import torch.nn.functional as F
from torch import nn

def inference(image_path, audio_path, text, model_path):
    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    # derived from ESResNeXt
    SAMPLE_RATE = 44100
    # derived from CLIP
    IMAGE_SIZE = 224
    IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
    IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
    
    aclp = AudioCLIP(pretrained=f'AudioCLIP-master/assets/{MODEL_FILENAME}')
    
    audio_transforms = ToTensor1D()
    
    image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])

    
    images = []
    audio = []
    texts = []
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    images.append(img)
    
    with torch.no_grad():
        track, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)
        
        # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
        # thus, the actual time-frequency representation will be visualized
        spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
        spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
        pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
        
        audio.append((track, pow_spec))
    
    texts.append(text)    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_audioclip(audio, images, texts): 
        
        
        with torch.no_grad():
            # AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
            audio_tensors = [audio_transforms(track.reshape(1, -1)) for track, _ in audio]
            # max_length = max(t.size(1) for t in tensors)
            # padded_tensors = [F.pad(t, (0, max_length - t.size(1)), "constant", 0) for t in tensors]
            audio = torch.stack([audio_tensors[0], audio_tensors[0]])
            # standard channel-first shape [batch x channels x height x width]
            image_tensors = [image_transforms(image) for image in images]
            images = torch.stack([image_tensors[0], image_tensors[0]])
            # textual input is processed internally, so no need to transform it beforehand
            texts = [texts, texts]
    
    
            # AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
            # Features = Tuple[AudioFeatures, ImageFeatures, TextFeatures]
            # Logits = Tuple[AudioImageLogits, AudioTextLogits, ImageTextLogits]
            ((audio_features, _, _), _), _ = aclp(audio=audio)
            end_time = time.time()
            
            ((_, image_features, _), _), _ = aclp(image=images)
            ((_, _, text_features), _), _ = aclp(text=texts)
    
            end_time = time.time()
            
            audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
            image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
            text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)
    
        return torch.cat((audio_features, image_features, text_features), dim=1) #image_features#
    features_embedding = process_audioclip(audio, images, texts).to(device)
    features_embedding = features_embedding[0]
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class MLP1(nn.Module):
        def __init__(self, input_shape: int, output_shape: int):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=input_shape, out_features=output_shape)
            # self.layer_2 = nn.Linear(in_features=hidden_shape, out_features=output_shape)
            # self.relu = nn.ReLU()
            # self.dropout = nn.Dropout(p=0.5)
    
        def forward(self, x):
            x = self.layer_1(x)
            return x
    
    class MLP2(nn.Module):
        def __init__(self, input_shape: int, output_shape: int, hidden_shape: int, DROP_OUT):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=input_shape, out_features=hidden_shape)
            self.layer_2 = nn.Linear(in_features=hidden_shape, out_features=output_shape)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=DROP_OUT)
    
        def forward(self, x):
            x = self.relu(self.layer_1(x))
            x = self.dropout(x)
            x = self.layer_2(x)
            return x
    
    mlp1 = MLP1(input_shape=3072, output_shape=3)
    #mlp2 = MLP2(input_shape=512, output_shape=5)
    
    # Load trained weights for MLP models
    mlp1.load_state_dict(torch.load(model_path))
    mlp1 = mlp1.to(device)
    #mlp2.load_state_dict(torch.load("mlp2_weights.pth"))
    
    mlp1.eval()
    #mlp2.eval()
    
    mood_mapping = {
        0: 'Sad',
        1: 'Neutral', 
        2: 'Happy'
    }
    
    output = mlp1(features_embedding)
    mood_idx = torch.argmax(output).item()
    mood_name = mood_mapping.get(mood_idx)    
    print(f"Running time: {time.time() - start_time:.4f} seconds")
    return mood_name


if __name__ == "__main__":
    image_path = "/project/vsharan_1298/aajinbo/csci566/data/MELD/MELD_kaggle/MELD-RAW/MELD_processed/demo_20_2000ms/test/images/dia0_utt0.png"
    audio_path = "/project/vsharan_1298/aajinbo/csci566/data/MELD/MELD_kaggle/MELD-RAW/MELD_processed/demo_20_2000ms/test/audio/dia0_utt0.wav"
    text = "i am very happy"
    model_path = "models/MLP_1.pth"
    a = inference(image_path, audio_path, text, model_path)
    print(a)