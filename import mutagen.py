''' Script to get mean and minimun duration of songs.'''

from mutagen.mp3 import MP3
import pickle5 as pickle

sountracks9000 = pickle.load(open('/home/pdbanet/Vionlabs/datasets/sound6600.pkl', 'rb'))

dataframes = [sountracks9000[i:i+200] for i in range(0,sountracks9000.shape[0],200)]
total_duration = 0

train = dataframes[0:33]
min_length = 300

for item in sountracks9000.iterrows():
    audio_path = '/home/pdbanet/Vionlabs/datasets/soundtracks9000/' + str(item[1].id) + '.mp3'
    audio = MP3(audio_path)
    
    sountracks9000.at[item[1].name, 'duration'] = audio.info.length

print(sountracks9000)


for dataframe in train:
    for item in dataframe.iterrows():
        audio_path = '/home/pdbanet/Vionlabs/datasets/soundtracks9000/' + str(item[1].id) + '.mp3'
        audio = MP3(audio_path)
        
        dataframe.at[item[1].name, 'duration'] = audio.info.length
        # Check minimun length
        if audio.info.length < min_length:
            min_length = audio.info.length
            id = item[1].id

        total_duration = total_duration + audio.info.length
    print(total_duration)
    print('min_length', min_length,'song id', id)

mean = total_duration / 4000
print(mean)
