'''Transform original dataset results to -1, 1 range, from 0,1 range'''

import json
import os
import numpy as np
import pickle5 as pickle

# sountracks9000.to_pickle('/home/pdbanet/Vionlabs/datasets/sountracks9000_withapport_something.pkl')
soundtracks6600 = pickle.load(open('/dataset/sountracks9000_withapport_something.pkl', 'rb'))
dataframes = [soundtracks6600[i:i+200] for i in range(0,soundtracks6600.shape[0],200)]
df_test = dataframes[32]

for song in df_test.iterrows():
    soundtracks6600.at[song[1]['id'], 'valence_tags'] = np.interp(song[1].valence_tags, (0, 1), (-1, +1))
    soundtracks6600.at[song[1]['id'], 'arousal_tags'] = np.interp(song[1].arousal_tags, (0, 1), (-1, +1))
    soundtracks6600.at[song[1]['id'], 'dominance_tags'] = np.interp(song[1].dominance_tags, (0, 1), (-1, +1))



print('finito')