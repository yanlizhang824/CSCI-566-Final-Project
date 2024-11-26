import torch
import torchvision.transforms as transforms
from PIL import Image
from AudioCLIP.model import AudioCLIP
from AudioCLIP.utils.transforms import ToTensor1D
import librosa
import torch.nn as nn
import numpy as np

# Load the models
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

# Initialize models
aclp = AudioCLIP(pretrained='AudioCLIP/assets/AudioCLIP-Full-Training.pt')
# change the dimension accordingly

mlp1 = MLP1(input_shape=3072, output_shape=3)
#mlp2 = MLP2(input_shape=512, output_shape=5)

# Load trained weights for MLP models
mlp1.load_state_dict(torch.load("mlp1_weights.pth", map_location=torch.device('cpu')))
#mlp2.load_state_dict(torch.load("mlp2_weights.pth"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to the correct device
mlp1.to(device)
aclp.to(device)

mlp1.eval()
#mlp2.eval()

# Preprocessing functions
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    transform = ToTensor1D()
    return transform(y).unsqueeze(0)  

def preprocess_text(text):
    return aclp.encode_text([text])  # Assuming AudioCLIP has a text encoding function

# Inference function
def infer(image_path, audio_path, text, model_type="mlp1"):
    # Preprocess inputs
    image_embedding = aclp.encode_image(preprocess_image(image_path).to(device))
    audio_embedding = aclp.encode_audio(preprocess_audio(audio_path).to(device))
    text_embedding = preprocess_text(text).to(device)

    # Concatenate embeddings
    combined_embedding = torch.cat([image_embedding, audio_embedding, text_embedding], dim=1).to(device)

    # Select model
    if model_type == "mlp1":
        output = mlp1(combined_embedding)
    #elif model_type == "mlp2":
        #output = mlp2(combined_embedding)
    else:
        raise ValueError("Invalid model type. Choose 'mlp1' or 'mlp2'.")

    # Get mood category
    mood_category = torch.argmax(output, dim=1).item()
    return mood_category




# Example usage
if __name__ == "__main__":
    image_path = "output.jpg"
    audio_path = "output.wav"
    text = "i am very happy"

    mood = infer(image_path, audio_path, text, model_type="mlp1")
    print(f"Predicted mood category: {mood}")
