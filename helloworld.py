"""
Test ob Entwicklungsumgebung läuft und ob cuda läuft
"""
import torch
if torch.cuda.is_available():
    print("Cuda läuft")
else: 
    print("Cuda nicht verfügbar")

