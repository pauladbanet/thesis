'''Creates the json with the parameters to pass to the job.'''

import json
import argparse
import sys 
sys.path.append('../thesis')

# from imports import * 
# from model_cnn import * 

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--user_first_arg',
      default=0.001)
  args = parser.parse_args()
  return args


args = get_args()
print(args.user_first_arg)

# params = {'batch_size': 64,
#           'n_epochs': 1000,
#           'dense_units': 128,
#           'learning_rate': 0.001,
#           'optimizer': 'Adam',
#           'n_dense': 4,
#           }

# for n_dense in [4, 5]:

# start_training(GC_PATHS[0:4], GC_PATHS[5], 0.001)

# with open('params.txt', 'w') as outfile:
#     json.dump(params, outfile)

# gcloud ai-platform jobs submit training test3 --config=config.json --master-image-uri=gcr.io/paula-309109/vscode --module-name=json_main.py --region=us-central1

































