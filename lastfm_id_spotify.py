''' 
5.
Get Spotify ID from sountracks9000 and then download mp3.
'''

from __init__ import *

os.environ['SPOTIPY_CLIENT_ID'] = '106ea9f6f35647b5a6f0321a99723d5a'
os.environ['SPOTIPY_CLIENT_SECRET'] = '1a2c3bacae8745a8af9b61194dcec9fe'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/'

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

sountracks9000 = pickle.load(open(os.path.join(os.path.expanduser("~"),'Vionlabs/datasets/sountracks9000.pkl') , 'rb'))

sountracks9000 = sountracks9000.dropna()  # this is to check if there is not spotify_id

# sountracks1000 = sountracks9000.head(1001)

# sountracks1000.to_pickle(os.path.join(dir, 'code/sountracks1000.pkl'))
# sountracks1000 = pickle.load(open(os.path.join(dir, 'sountracks1000.pkl'), 'rb'))
# sountracks1000 = sountracks1000.dropna()
no_id = []

def get_spotify_id(item):
    query = item['artist'] + " " + item['track']
    query = query.replace("&", "")
    query = re.sub(r" ?\([^)]+\)", "", query)
    query = query.split("-")[0]
    track_results = spotify.search(q=query, type='track', limit=50)    
    
    if track_results['tracks']['total'] != 0:
        # From the results take the first one
        id = track_results['tracks']['items'][0]['id']
        audio_features = spotify.audio_features(id) 
        sountracks9000.at[item['id'], 'spotify_id'] = id
        sountracks9000.at[item['id'], 'duration'] = audio_features[0]['duration_ms']
        
        # print('duration: ' + str(audio_features[0]['duration_ms']))

    else:
        print(item['id'], query)
        no_id.append(item['id'])


def mp3(item):
    link = "https://open.spotify.com/track/" +  str(item.spotify_id)

    dir = os.getcwd()
    subprocess.call([os.path.join(dir, 'shell.sh'), link])

    list_of_files = glob.glob('/home/pdbanet/.local/share/Savify/downloads/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    new_file_name = "/home/pdbanet/Vionlabs/datasets/soundtracks9000/" + str(item['id']) + ".mp3"
    os.rename(latest_file, new_file_name)

    audio = MP3(new_file_name)
    sountracks9000.at[item['id'], 'duration'] = audio.info.length


sountracks9000.apply(lambda row: mp3(row) if row.name >= 1154 else 0, axis = 1)

