import os
import torch
from torch.nn.modules import fold
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from music_classifier.dataset import load_audio_dir, load_joint_audio_dir
from ml_logger import logger


class InstrumentDataset(Dataset):
    def __init__(self, directory, patch_length=262144, clip_length=12.0, sample_rate=22050):
        """
        Args:
            directory (string): Directory with all the WAV files.
            patch_length (int): Length of each audio patch in samples.
            clip_length (float): Length of each audio clip in seconds.
            sample_rate (int): Sample rate of the audio files.
        """
        self.patch_length = patch_length
        self.directory = directory
        self.clip_length = clip_length
        self.sample_rate = sample_rate
        self.songs = []

        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                instrument_files = ['bass.wav', 'drums.wav', 'guitar.wav', 'piano.wav']
                tracks = []
                min_stem_length = None
                for instrument_file in instrument_files:
                    filepath = os.path.join(subdir_path, instrument_file)
                    if os.path.exists(filepath):
                        waveform, _ = torchaudio.load(filepath, format='wav')
                        tracks.append(waveform)
                        min_stem_length = waveform.shape[1] if min_stem_length is None else min(min_stem_length,
                                                                                                waveform.shape[1])

                if len(tracks) == 4 and min_stem_length >= clip_length * sample_rate:
                    for i in range(len(tracks)):
                        tracks[i] = tracks[i][:, :min_stem_length]
                    song_tensor = torch.zeros((4, min_stem_length))
                    for i in range(len(tracks)):
                        song_tensor[i, :] = tracks[i][0, :]
                    self.songs.append(song_tensor)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        waveform = self.songs[idx][:, :]
        start_sample = np.random.randint(0, waveform.shape[1] - self.patch_length)
        end_sample = start_sample + self.patch_length
        return waveform[:, start_sample:end_sample]


class RNNMultiDataset(Dataset):
    def __init__(self, directory, clips, patch_length=131072, clip_length=6, sample_rate=22050):
        """
        Args:
            directory (string): Directory with all the WAV files.
            clip_length (int): Length of each audio clip in seconds.
            sample_rate (int): Sample rate of the audio files.
        """
        self.directory = directory
        self.patch_length = patch_length
        self.clip_length = clip_length
        self.sample_rate = sample_rate
        self.clips = clips
        self.songs = []

        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                instrument_files = ['bass.wav', 'drums.wav', 'guitar.wav', 'piano.wav']
                tracks = []
                min_stem_length = None
                for instrument_file in instrument_files:
                    filepath = os.path.join(subdir_path, instrument_file)
                    if os.path.exists(filepath):
                        waveform, _ = torchaudio.load(filepath, format='wav')
                        tracks.append(waveform)
                        min_stem_length = waveform.shape[1] if min_stem_length is None else min(min_stem_length,
                                                                                                waveform.shape[1])

                if len(tracks) == 4 and min_stem_length >= clips * clip_length * sample_rate:
                    for i in range(len(tracks)):
                        tracks[i] = tracks[i][:, :min_stem_length]
                    song_tensor = torch.zeros((4, min_stem_length))
                    for i in range(len(tracks)):
                        song_tensor[i, :] = tracks[i][0, :]
                    self.songs.append(song_tensor)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        waveform = self.songs[idx][:, :]
        start_sample = np.random.randint(0, waveform.shape[1] - self.clips * self.patch_length)
        end_sample = start_sample + self.clips * self.patch_length
        return waveform[:, start_sample:end_sample]


class RNNDataset(Dataset):
    def __init__(self, directory, min_clips=3, clip_length=12, sample_rate=22050):
        """
        Args:
            directory (string): Directory with all the WAV files.
            clip_length (int): Length of each audio clip in seconds.
            sample_rate (int): Sample rate of the audio files.
        """
        self.directory = directory
        self.clip_length = clip_length
        self.sample_rate = sample_rate
        self.min_clips = min_clips
        self.songs = []
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                waveform, _ = torchaudio.load(filepath, format='wav')
                num_clips = waveform.shape[1] // (clip_length * sample_rate)
                if num_clips >= min_clips:
                    self.songs.append(waveform)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        return self.songs[idx]


class FullSongDataset(Dataset):
    def __init__(self, directory, target_length, folder_with_wavs):
        self.songs = load_joint_audio_dir(directory) if folder_with_wavs else \
                     load_audio_dir(directory, 0)
        self.songs = [song for song in self.songs if song.shape[1] == target_length]
        logger.print(f'Dataset size: {len(self.songs)}')

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        return self.songs[idx]
