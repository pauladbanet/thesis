import pickle5 as pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os

# sountracks9000.to_pickle('/home/pdbanet/Vionlabs/datasets/sountracks9000_newweights.pkl')
# sountracks9000.to_pickle('/home/pdbanet/Vionlabs/datasets/sountracks9000_withapport_something.pkl')
sountracks9000 = pickle.load(open('/home/pdbanet/Vionlabs/datasets/sountracks9000.pkl', 'rb'))
nrc = pd.read_csv('/home/pdbanet/Vionlabs/datasets/csv_vad/NRC.csv')

remove_tags = ['soundtrack', 'electronic', 'indie', 'rock', 'movie',
                'pop', 'indie', 'country', 'instrumental', 'classical',
                'classic', 'alternative', 'rap', 'industrial', 'musical',
                'classic rock', 'acoustic', 'game', 'jazzy', 'jazz', 'ambient',
                'folk', 'cover', 'blues', 'country', 'video game', 'metal', 'funk', 
                'house', 'world', 'new', 'vocal', 'composer', 'film', 'intro', 'television',
                'duet', 'chorus', 'music', 'movies', 'tag']

aport_something = ['soul', 'heavy metal', 'oldies', 'punk', 'emo', 'experimental', 'mariachi',
        'dance', 'piano', 'guitar', 'crunchy', 'hip hop', 'techno', 'disco', 'reggae', 'trumpet',
        'orchestra', 'new age', 'hippie', 'violin', 'funky', 'gothic', 'hardcore', 'western']

def remove_weights(row):
    new_weights = []

    for mood, value in row[1].weighted_mood.items():
        if mood in remove_tags:
            new_weights.append(mood)
        elif mood in aport_something:
            new_weights.append(mood)

    for i in new_weights:
        del row[1].weighted_mood[i]


print('BEFORE', sountracks9000.head(10))
for row in sountracks9000.iterrows():
    remove_weights(row)

print('AFTER', sountracks9000.head(10))


def new_average(row):
    valence = []
    weights = []
    arousal = []
    dominance = []
    weighted_mood = {}

    def check_nrc(mood, value):
        aux_name = nrc.loc[nrc['Word'] == mood.lower(), :]
        if aux_name.empty == False:

            aux = aux_name.to_numpy()
            weighted_mood[aux[0][0]] = value
            valence.append(aux[0][1] * value)
            arousal.append(aux[0][2] * value)
            dominance.append(aux[0][3] * value)
            weights.append(value)     

    for mood, value in row[1].weighted_mood.items():
        check_nrc(mood, value)
        print(row[1]['id'])

    if len(weighted_mood) != 0:
        # Make the average VAD and place it on df.
        sountracks9000.at[row[1]['id'], 'id'] = row[1]['id']
        sountracks9000.at[row[1]['id'], 'weighted_mood'] = weighted_mood
        sountracks9000.at[row[1]['id'], 'valence_tags'] = sum(valence) / sum(weights)
        sountracks9000.at[row[1]['id'], 'arousal_tags'] = sum(arousal) / sum(weights)
        sountracks9000.at[row[1]['id'], 'dominance_tags'] = sum(dominance) / sum(weights)


for row in sountracks9000.iterrows():
    new_average(row) 


print('aver')
# response = sountracks9000.apply(lambda row : new_average(row), axis = 1)
