import torch as t
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Dispositivo detectado para o treino: {device}")
if device.type == "cuda":
    print(f"Placa gráfica: {t.cuda.get_device_name(0)}")