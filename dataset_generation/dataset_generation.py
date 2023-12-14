import torchaudio
import torch
from torch import Tensor
from librosa.beat import beat_track
import pathlib
import h5py

MIN_TEMPO = 135
SAMPLE_RATE = 44_000
K_BEATS = 16
NUM_SAMPLES = int(SAMPLE_RATE * K_BEATS // (MIN_TEMPO / 60))

PATH = pathlib.Path.home() / "Documents" / "Music" / "Samoh"
PATH = PATH if PATH.is_dir() else "ONLINE_PATH"

def resample(signal: Tensor, sr: int) -> Tensor:
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        signal = resampler(signal)
    return signal


def mix_down(signal: Tensor) -> Tensor:
    # aggregates different audio channels
    if signal.shape[0] > 1:
        # (n_channels, n_samples) -> (1, n_samples)
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def get_beat_locations(signal: Tensor, sr: int = SAMPLE_RATE, verbose: bool = False):
    tempo, beats = beat_track(y=signal.numpy().reshape(-1,), sr=sr, start_bpm=140.0, units='samples')
    if verbose:
        print("Estimated Tempo of current track is {}".format(tempo))
    return tempo, beats.tolist()


def create_sequences(signal: Tensor, beat_locations, k_beats=K_BEATS, verbose:bool=False) -> Tensor:        
    seq_list = [] # to test TODO: move it to seperate test file where I don't need to pad
    padded_seq_list = []

    old_beat_idx = 0
    for beat_idx in range(1, len(beat_locations)):
        if beat_idx % k_beats == 0:
            sequence = signal[:, beat_locations[old_beat_idx] : beat_locations[beat_idx]]
            seq_list.append(sequence)
            sequence = apply_padding(sequence, verbose=verbose)
            padded_seq_list.append(sequence)
            old_beat_idx = beat_idx


    # we igonore all signals 
    start = signal[:, : beat_locations[0] ] # before first beat
    ending = signal[:, beat_locations[old_beat_idx] : ] # after last beat
    
    # testing
    expected_count = start.shape[1] + sum(map(lambda x: x.shape[1], seq_list)) + ending.shape[1]
    assert expected_count == signal.shape[1]
    
    padded_track = torch.cat(padded_seq_list, dim=0) # (len(padded_seq_list) ,sigal.shape[1])
    if verbose:
        print("Track successfully secences with shape: {0}".format(padded_track.shape))

    return padded_track 


def apply_padding(signal:Tensor, verbose:bool=False) -> Tensor:
    n_missing_samples = NUM_SAMPLES - signal.shape[1]
    if n_missing_samples < 0:
        # cut sample with warning
        signal = signal[:, :NUM_SAMPLES]
        if verbose:
            print("Warning: One Signal had too many samples and was cut!")
    
    if n_missing_samples > 0:
        # rigth padding
        dims = (0, n_missing_samples)
        signal = torch.nn.functional.pad(signal, dims)
    return signal


if __name__ == "__main__":
    # TODO: add parser to debugging + add debug arugments in fucntion
    verbose = False
    
    
    
    song_paths = [str(song_file) for song_file in list(PATH.glob("*.wav"))]
    
    features = {}
        
    for idx, song_path in enumerate(song_paths):
        # Loading and preparing track (audio file)
        audio_wave, sr = torchaudio.load(song_path)
        audio_wave = resample(audio_wave, sr)
        audio_wave = mix_down(audio_wave)
        
        # Splitting track by beats 
        tempo, beat_locations = get_beat_locations(audio_wave, verbose=verbose)
        
        
        # ignore slow tracks (slice sizes would be too long)
        if tempo >= MIN_TEMPO: 
            # sequence audio file
            audio_seq = create_sequences(audio_wave, beat_locations, verbose=verbose)
            features[str(idx)] = audio_seq
    
    # store data in hdf5 format in data directory
    data_path = pathlib.Path(__file__).parent / "data"
    data_path.mkdir(exist_ok=True)
    
    f=h5py.File(data_path / f"techno_{SAMPLE_RATE}_{K_BEATS}.h5",'w')    
    grp=f.create_group('TechnoGen')
    for idx, track in features.items():
        grp.create_dataset(idx, data=track.numpy())
                
    f.close()