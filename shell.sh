#!/bin/bash
sudo docker run --rm -v "/home/pdbanet/.local/share/Savify/downloads:/root/.local/share/Savify/downloads" \
-e SPOTIPY_CLIENT_ID='106ea9f6f35647b5a6f0321a99723d5a' \
-e SPOTIPY_CLIENT_SECRET='1a2c3bacae8745a8af9b61194dcec9fe' \
laurencerawlings/savify:latest $1
