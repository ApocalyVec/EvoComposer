from utils.MIDI_utils import read_midi, prepare_xy, prepare_x,create_filtered, encode_seq
file = '/Users/liy/Downloads/evocomposer/schubert/schubert_D850_1.mid'

notes, sampled, freq_sample = read_midi(file)