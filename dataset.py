import os
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset


class ProjectAudioCommandDataset(Dataset):
    """Command Audio dataset"""

    def __init__(self, root_dir='./', transformation=None, subset: str = None):
        """
        Args:
            csv_file (string): Path to the csv file
            root_dir (string): Directory with all the audio files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.mode = subset
        if subset == 'training' or subset == 'validation':
            self.anno = pd.read_csv('./dsl_data/development.csv')
            self.anno['command'] = self.anno['action'] + (self.anno['object'] != 'none') * self.anno['object']
            self.classes = self.anno['command'].unique().tolist()
            self.number_of_classes = len(self.classes)
            self.cut_time = 5  # seconds
        if subset == 'testing':
            self.anno = pd.read_csv('./dsl_data/evaluation.csv')
            self.cut_time = 5  # seconds

        self.root_dir = root_dir
        self.transform = transformation

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.mode == 'training' or self.mode == 'validation':
            audio_path = os.path.join(self.root_dir,
                                      self.anno.iloc[idx, 1])
            sample_rate, waveform = wavfile.read(audio_path)
            duration = len(waveform) / sample_rate

            if duration >= self.cut_time:
                new_len = sample_rate * self.cut_time
                waveform = waveform[0:new_len]
            elif duration < self.cut_time:
                new_len = sample_rate * self.cut_time
                waveform = np.concatenate([waveform, np.zeros(new_len - len(waveform))], axis=0)

            label = self.anno.iloc[idx]['command']
            speaker_id = self.anno.iloc[idx]['speakerId']

            if self.transform:
                waveform = self.transform(waveform)

            return torch.tensor(waveform, dtype=torch.float32).unsqueeze(dim=0), sample_rate, label, speaker_id
        
        elif self.mode == 'testing':
            
            audio_path = os.path.join(self.root_dir,
                                      self.anno.iloc[idx, 1])
            sample_rate, waveform = wavfile.read(audio_path)
            duration = len(waveform) / sample_rate

            if duration > self.cut_time:
                new_len = sample_rate * self.cut_time
                waveform = waveform[0:new_len]
            elif duration < self.cut_time:
                new_len = sample_rate * self.cut_time
                waveform = np.concatenate([waveform, np.zeros(new_len - len(waveform))], axis=0)

            speaker_id = self.anno.iloc[idx]['speakerId']
            file_id = self.anno.iloc[idx]['Id']
            if self.transform:
                waveform = self.transform(waveform)

            return file_id, torch.tensor(waveform, dtype=torch.float32).unsqueeze(dim=0), sample_rate, speaker_id
