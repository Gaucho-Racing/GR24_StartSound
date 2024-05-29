import torch
import torchaudio
import tkinter as tk
from tkinter import filedialog

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getLocalFile():
    root=tk.Tk()
    root.withdraw()
    filePath=filedialog.askopenfilename(title = "Select music file to compress")
    print('file path: ',filePath)
    return filePath

#read audio file
audio, _ = torchaudio.load(getLocalFile(), normalize=True, channels_first=True)
audio = audio[0]
print(audio.shape, audio.dtype)

#normalize to 0 ~ 255
audio = audio.to(device)
audio /= torch.max(torch.abs(audio))
audio += 1
audio *= 127
print(audio.max(), audio.min());

#convert to uint_8t
audio = audio.type(torch.uint8)
print(audio.max(), audio.min());
audioList = audio.tolist()
print(str(audioList)[1:-1])
print(len(audioList))
