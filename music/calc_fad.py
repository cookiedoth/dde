import argparse
import os

import librosa
import soundfile as sf
from frechet_audio_distance import FrechetAudioDistance


def resample_and_cut_wav_files(source_dir, target_dir, target_sample_rate, max_length):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(source_dir, filename)
            audio, sr = librosa.load(file_path, sr=22050)
            duration = librosa.get_duration(y=audio, sr=22050)

            if duration > max_length:
                max_length_samples = int(max_length * 22050)
                start_sample = (len(audio) - max_length_samples) // 2
                audio = audio[start_sample:start_sample + max_length_samples]

            audio_resampled = librosa.resample(audio, orig_sr=22050, target_sr=target_sample_rate)
            output_file_path = os.path.join(target_dir, filename)
            sf.write(output_file_path, audio_resampled, target_sample_rate)
            print(f"Resampled and saved: {output_file_path}")


def main(dataset_path, eval_path, max_length):
    tmp_dataset_path = "-"
    tmp_eval_path = "-"

    resample_and_cut_wav_files(dataset_path, tmp_dataset_path, 16000, max_length)
    resample_and_cut_wav_files(eval_path, tmp_eval_path, 16000, max_length)

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False
    )

    fad_score = frechet.score(tmp_dataset_path, tmp_eval_path, dtype="float32")
    print(f"FAD score: {fad_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and max length value.')
    parser.add_argument('dataset_path', type=str, help='The path to the dataset directory')
    parser.add_argument('eval_path', type=str, help='The path to the sampled tracks directory')
    parser.add_argument('max_length', type=float, help='Maximum length of the track in seconds')

    args = parser.parse_args()
    main(args.dataset_path, args.eval_path, args.max_length)
