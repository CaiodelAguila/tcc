import numpy as np
import torch as t
import torch.nn as nn
import audio_cnn as ac
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader

class SpeechCommandsDataset(Dataset):
    def __init__(self, lista_caminhos,lista_labels,label_idx):
        self.caminhos = lista_caminhos
        self.labels = lista_labels
        self.label_idx = label_idx

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        caminho = self.caminhos[idx]
        label = self.labels[idx]
        y, _ = librosa.load(caminho, sr=16000)
        if(len(y) != 16000):
            y = librosa.util.fix_length(y, size=16000)
        
        tensor_audio = t.tensor(y, dtype=t.float32)
        label_idx = self.label_idx[label]
        tensor_label = t.tensor(label_idx, dtype=t.long)
        return tensor_audio, tensor_label

caminho_dataset = r"E:\TCC\dataset\SpeechCommands\speech_commands_v0.02"

caminho_validation = os.path.join(caminho_dataset, "validation_list.txt")
caminho_testing = os.path.join(caminho_dataset, "testing_list.txt")

with open(caminho_validation, 'r') as f:
    caminhos_validation = set(line.strip() for line in f.readlines())
with open(caminho_testing, 'r') as f:
    caminhos_testing = set(line.strip() for line in f.readlines())

classes_alvo = [pasta for pasta in os.listdir(caminho_dataset) 
                if os.path.isdir(os.path.join(caminho_dataset, pasta)) 
                and pasta != "_background_noise_"]

caminhos_treino,labels_treino = [],[]
caminho_validation,labels_validation = [],[]
caminho_testing,labels_testing = [],[]

for classe in classes_alvo:
    caminho_classe = os.path.join(caminho_dataset, classe)
    for arquivo in os.listdir(caminho_classe):
        if(arquivo.endswith(".wav")):
            caminho_relativo = os.path.join(classe, arquivo)
            if caminho_relativo in caminhos_validation:
                caminho_validation.append(os.path.join(caminho_classe, arquivo))
                labels_validation.append(classe)
            elif caminho_relativo in caminhos_testing:
                caminho_testing.append(os.path.join(caminho_classe, arquivo))
                labels_testing.append(classe)
            else:
                caminhos_treino.append(os.path.join(caminho_classe, arquivo))
                labels_treino.append(classe)

classes_unicas = sorted(list(set(labels_treino)))

label_to_idx = {label: idx for idx, label in enumerate(classes_unicas)}

dataset_treino = SpeechCommandsDataset(caminhos_treino, labels_treino, label_to_idx)

dataset_loader = DataLoader(dataset_treino, batch_size=32, shuffle=True)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
modelo = ac.AudioNet1D(num_classes=35)
modelo = modelo.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(modelo.parameters(), lr=0.001)

num_age = 3

for epoch in range(num_age):
    modelo.train()
    running_loss = 0.0


    for lote_idx, (audios, labels) in enumerate(dataset_loader):
        audios = audios.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = modelo(audios.unsqueeze(1))

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if (lote_idx + 1) % 100 == 0:
            print(f"Época [{epoch+1}/{num_age}], Lote [{lote_idx+1}/{len(dataset_loader)}], Erro atual: {loss.item():.4f}")

    erro_medio = running_loss / len(dataset_loader)
    print(f"Época {epoch+1}/{num_age}, Loss: {erro_medio}")