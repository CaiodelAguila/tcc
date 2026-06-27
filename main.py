import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import audio_cnn as ac
import numpy as np
import librosa
import os
import time
from torch.utils.data import Dataset, DataLoader
import audio_processing as ap
from tqdm import tqdm

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

class SpeechCommandsDatasetV2(Dataset):
    def __init__(self, lista_caminhos,lista_labels,label_idx):
        self.caminhos = lista_caminhos
        self.labels = lista_labels
        self.label_idx = label_idx
        
        self.frequencies = np.linspace(20.0, 8000.0 , 64) # Frequências de 0 a 8kHz, com 64 pontos
        #self.frequencies = np.geomspace(20.0, 8000.0, num=64)  # Frequências de 20Hz a 8kHz, com 64 pontos
        self.compress = nn.AdaptiveAvgPool1d(256)
    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        caminho = self.caminhos[idx]
        label = self.labels[idx]

        audio = ap.carregar_audio(caminho, sample_rate=16000)

        features = ap.extract_features(audio, self.frequencies)

        tensor_audio = t.tensor(features, dtype=t.float32)
        tensor_audio = self.compress(tensor_audio)

        label_idx = self.label_idx[label]
        tensor_label = t.tensor(label_idx, dtype=t.long)


        return tensor_audio, tensor_label

caminho_dataset = r"E:\TCC\dataset\SpeechCommands\speech_commands_v0.02"

caminho_validation = os.path.join(caminho_dataset, "validation_list.txt")
caminho_testing = os.path.join(caminho_dataset, "testing_list.txt")

with open(caminho_validation, 'r', encoding='utf-8') as f:
    caminhos_validation = set(line.strip() for line in f.readlines())
with open(caminho_testing, 'r', encoding='utf-8') as f:
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
            caminho_relativo = f"{classe}/{arquivo}"
            if caminho_relativo in caminhos_validation:
                caminho_validation.append(os.path.join(caminho_classe, arquivo))
                labels_validation.append(classe)
            elif caminho_relativo in caminhos_testing:
                caminho_testing.append(os.path.join(caminho_classe, arquivo))
                labels_testing.append(classe)
            else:
                caminhos_treino.append(os.path.join(caminho_classe, arquivo))
                labels_treino.append(classe)

print("-"*40)
print("Número de amostras no conjunto de treino:", len(caminhos_treino))
print("Número de amostras no conjunto de validação:", len(caminho_validation))
print("Número de amostras no conjunto de teste:", len(caminho_testing))
print("-"*40)

classes_unicas = sorted(list(set(labels_treino)))

label_to_idx = {label: idx for idx, label in enumerate(classes_unicas)}
#Escolha do dataset para treino e teste, utilizando a versão 2 do dataset, que aplica a transformada de Morlet e compressão
#dataset_treino = SpeechCommandsDataset(caminhos_treino, labels_treino, label_to_idx) #V1
dataset_treino = SpeechCommandsDatasetV2(caminhos_treino, labels_treino, label_to_idx) #V2
dataloader_treino = DataLoader(dataset_treino, batch_size=128, shuffle=True,num_workers=4)

#dataset_teste = SpeechCommandsDataset(caminho_testing, labels_testing, label_to_idx) #V1
dataset_teste = SpeechCommandsDatasetV2(caminho_testing, labels_testing, label_to_idx) #V2
dataloader_teste = DataLoader(dataset_teste, batch_size=128, shuffle=False,num_workers=4)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
#Escolha da audionet para o modelo
#modelo = ac.AudioNet1D(num_classes=35)  #V1
modelo = ac.AudioNet1DV2(num_classes=35) #V2
modelo = modelo.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = t.optim.Adam(modelo.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer = t.optim.Adam(modelo.parameters(), lr=0.001)

#checkpoint_path = "checkpoint_audionet.pth" #V1
checkpoint_path = "checkpoint_audionetv2.pth" #V2
start=0
if(os.path.exists(checkpoint_path)):
    checkpoint = t.load(checkpoint_path)
    modelo.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    start = checkpoint['epoch']

num_age = 2
erro_treino = []
acc_teste = []

start_time = time.perf_counter()
for epoch in range(start,num_age):
    modelo.train()
    running_loss = 0.0
    print(f"Epoca : {epoch+1}")

    for lote_idx, (audios, labels) in enumerate(tqdm(dataloader_treino,desc="Training")):
        audios = audios.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #outputs = modelo(audios.unsqueeze(1)) #V1
        outputs = modelo(audios) #V2
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    erro_medio = running_loss / len(dataloader_treino)
    erro_treino.append(erro_medio)
    
    modelo.eval()
    acertos = 0
    total = 0
    with t.no_grad():
        for audios, labels in tqdm(dataloader_teste):
            audios = audios.to(device)
            labels = labels.to(device)

            #outputs = modelo(audios.unsqueeze(1)) #V1
            outputs = modelo(audios) #V2
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            acertos += (predicted == labels).sum().item()
    acuracia = 100 * acertos / total
    acc_teste.append(acuracia)
    checkpoint ={'epoch' : epoch,'model_state' : modelo.state_dict(),'optim_state' : optimizer.state_dict()}
    
    #t.save(checkpoint,"checkpoint_audionet.pth") #V1
    t.save(checkpoint,"checkpoint_audionetv2.pth") #V2
    print("[CHECKPOINT]")
end_time=time.perf_counter()

print(f"Tempo de treinamento: {end_time-start_time:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_age + 1), erro_treino, marker='o', color='red', label='Loss de Treino')
plt.title('Evolução do Erro (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss (CrossEntropy)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_age + 1), acc_teste, marker='o', color='blue', label='Acurácia de Teste')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()