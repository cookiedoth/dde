import os
import torchaudio


def main(source_directory, target_directory, clip_length=12, sample_rate=22050):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for subdir in os.listdir(source_directory):
        subdir_path = os.path.join(source_directory, subdir)
        if os.path.isdir(subdir_path):
            instrument_files = ['bass.wav', 'drums.wav', 'guitar.wav', 'piano.wav']
            tracks = []
            min_stem_length = None
            for instrument_file in instrument_files:
                filepath = os.path.join(subdir_path, instrument_file)
                if os.path.exists(filepath):
                    waveform, _ = torchaudio.load(filepath, format='wav')
                    tracks.append(waveform)
                    if min_stem_length is None or waveform.shape[1] < min_stem_length:
                        min_stem_length = waveform.shape[1]

            if len(tracks) == 4 and min_stem_length >= clip_length * sample_rate:
                for i in range(len(tracks)):
                    tracks[i] = tracks[i][:, :min_stem_length]
                combined_waveform = sum(tracks)
                combined_filename = f"{subdir}_combined.wav"
                combined_filepath = os.path.join(target_directory, combined_filename)
                torchaudio.save(combined_filepath, combined_waveform, sample_rate)


if __name__ == "__main__":
    # Creates a train dataset for 4 instrument average songs.
    source_directory = "/data/scratch/diff-comp/audio/slakh2100/train"
    target_directory = "/data/scratch/diff-comp/merged"
    main(source_directory, target_directory)