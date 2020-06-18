from utils.MIDI_utils import generate_MIDI_representation
import matplotlib.pyplot as plt
import numpy as np

file = '/Users/Leo/PycharmProjects/EvoComposer/data/schubert/schubert_D850_1.mid'
release_freq = 0
rest_freq = -1

notes, sampled, sampled_freq = generate_MIDI_representation(file, release_freq, rest_freq)

# plot a 128 segment
release = np.empty(shape=(0, 2))
rest = np.empty(shape=(0, 2))
pitch = np.empty(shape=(0, 2))


# non interactive
# interval = slice(68700, 69000)
# for i, f in enumerate(sampled_freq[interval]):
#     print((i, f))
#     s = np.array([[i, f]])
#     if f == release_freq:
#         release = np.concatenate((release, s))
#     elif f == rest_freq:
#         rest = np.concatenate((rest, s))
#     else:
#         pitch = np.concatenate((pitch, s))

# fig, ax = plt.subplots()
# fig.set_size_inches(30, 8)
# ax.scatter(pitch[:, 0], pitch[:, 1], marker='o', label='Pitch events')
# ax.scatter(rest[:, 0], rest[:, 1], marker='x', label='Rest events')
# ax.scatter(release[:, 0], release[:, 1], marker='^', label='Release events')
# ax.set_title('Representation of a segment in schubert_D850_1')
# ax.set_xlabel('Time steps')
# ax.set_ylabel('Frequency (Hz)')
# ax.legend()
# plt.show()

# interactive mode
for i, f in enumerate(sampled_freq):
    s = np.array([[i, f]])
    if f == release_freq:
        release = np.concatenate((release, s))
    elif f == rest_freq:
        rest = np.concatenate((rest, s))
    else:
        pitch = np.concatenate((pitch, s))

plt.scatter(pitch[:, 0], pitch[:, 1], marker='o', label='Pitch events')
plt.scatter(rest[:, 0], rest[:, 1], marker='x', label='Rest events')
plt.scatter(release[:, 0], release[:, 1], marker='^', label='Release events')
plt.ylabel('Time steps')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.show()