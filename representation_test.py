from utils.MIDI_utils import generate_MIDI_representation
import matplotlib.pyplot as plt

file = '/Users/Leo/PycharmProjects/EvoComposer/data/schubert/schubert_D850_1.mid'

notes, sampled, sampled_freq = generate_MIDI_representation(file)

# plot a 128 segment
plt.scatter(list(range(len(sampled_freq[7500:8127]))), sampled_freq[7500:8127])
plt.show()