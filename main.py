from fastapi import FastAPI, File, UploadFile

import torchaudio
import torch
from model import AudioLSTM
import numpy as np

app = FastAPI()


@app.get('/')
def index():
    return {'data':{'author':'Thiago', 'description': 'FastAPI and audio classification'}}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    unique_label = {3: 'dog_bark',2: 'children_playing',1: 'car_horn',0: 'air_conditioner',9: 'street_music',6: 'gun_shot', 8: 'siren', 5: 'engine_idling',7: 'jackhammer',4: 'drilling'}


    waveform, sr = torchaudio.load(file.file)

    audio_mono = torch.mean(waveform, dim=0, keepdim=True)
    temp_data = torch.zeros([1, 160000])
    if audio_mono.numel() < 160000:
        temp_data[:, :audio_mono.numel()] = audio_mono
    else:
        temp_data = audio_mono[:, :160000]
        
    audio_mono=temp_data

    mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)

    mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)

    new_feat = torch.cat([mel_specgram, mfcc], axis=1)

    data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
    saved = torch.load("./mymodel.tar", map_location=torch.device("cpu"))["state_dict"]
    model = AudioLSTM(n_feature=168, out_feature=len(unique_label))

    model.load_state_dict(saved)

    with torch.no_grad():
        for x in data:
            output, hidden_state = model(x, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
            max_v = np.argmax(output.numpy())
            for i, v in unique_label.items():
                if  max_v == i:
                    return {"predicted": v}
    return {"Error": file.filename}