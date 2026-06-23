import os
import torchaudio

folder = "dataset/"

os.makedirs(folder, exist_ok=True)

dataset_treino = torchaudio.datasets.SPEECHCOMMANDS(url='speech_commands_v0.02',root=folder, download=True, subset="training")

print("Número de amostras no conjunto de treino:", len(dataset_treino))