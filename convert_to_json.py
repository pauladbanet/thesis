''' Script to convert pickle to json.'''

import json
import pandas
import pickle5 as pickle

with open('sountracks9000_withapport_something.jsonl', 'w') as f:
    x = pandas.read_pickle(open('/dataset/sountracks9000_withapport_something.pkl', 'rb'))
    for (name, data) in x.iterrows():
        json_item = {
            '_id': data.id,
            'title': data.artist + ' - ' + data.track,
            'weighted_mood': data.weighted_mood,
            'spotify_id': data.spotify_id,
            'tags': {
                'valence_tags': data.valence_tags,
                'arousal_tags': data.arousal_tags,
                'dominance_tags': data.dominance_tags,
            },
        }
        f.write(json.dumps(json_item) + '\n')


