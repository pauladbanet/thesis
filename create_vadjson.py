import json
from google.cloud import storage
import pickle5 as pickle
import numpy as np

soundtracks6600 = pickle.load(open('/dataset/soundtracks6600.pkl', 'rb'))
dataframes = [soundtracks6600[i:i+200] for i in range(0,soundtracks6600.shape[0],200)]
df_test = dataframes[32]

storage_client = storage.Client('paula-309109')
bucket = storage_client.get_bucket('job_results')

blob_val = bucket.blob('lstm_vgg_15_valence/val_predicted.json')
val = json.loads(blob_val.download_as_string(client=None))

blob_aro = bucket.blob('lstm_vgg_15_aro/aro_predicted.json')
aro = json.loads(blob_aro.download_as_string(client=None))

blob_dom = bucket.blob('lstm_vgg_15_dom/dom_predicted.json')
dom = json.loads(blob_dom.download_as_string(client=None))

print('len(val)',len(val))
print('len(aro)',len(aro))
print('len(dom)',len(dom))

songs_id = []
for song in df_test.iterrows():
    songs_id.append(song[1].id)

for v, a, d, id in zip(val, aro, dom, songs_id):
    v = map(float, v)
    v = np.array(list(v))
    a = map(float, a)
    a = np.array(list(a))
    d = map(float, d)
    d = np.array(list(d))
    vad_song = np.array([v, a, d]).transpose()
    times = np.arange(0, len(v)*150000, step=150000)
    jotason = {"data": vad_song.tolist(),
    "type":"vionfeatures.models.audio.VADAudio",
    "_version":"2.1.4",
    "timestamps": times.tolist(),
    "dataVersion":"1.0.4"}
    print(id)

    with open('/dataset/test_32/' + str(id) + '_pred.json', 'w') as f:
        f.write(json.dumps(jotason))

print('finito')