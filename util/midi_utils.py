from __future__ import division
from .write_midi import *


def save_midis(bars, file_path, tempo=80.0, midi_shape=(64, 84)):
    bars = np.resize(bars, midi_shape)
    bars = bars.reshape(1, bars.shape[0], bars.shape[1], 1)
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))
    write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
                              tempo=tempo, beat_resolution=4)
