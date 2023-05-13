# nnusic
Artificial Intelligence way to compose the music needed to focus, relax or sleep

# FAQ

## What is the main idea behind this project, why is it needed?
> This project is a simple application for music generation.
> You will be able to compose accompaniment for your future song, a new motif, or a vibe music for focus/relaxation/sport/sleeping/etc.
> The main goal is to replace expensive pay-to-play music services, such as brain fm and others. 
> No offense to those services, they are exellent, but some users do not have a possibility to pay. nnusic will solve this problem for some users.

## What is completed so far?
> A simple algorithm for music generation using a classical LSTM from Keras

## What stage this project is on currently?
> The project was started in the beginning of March 2023.
> At the moment it is on it's first MVP (generation of simple piano tracks using LSTM and small dataset to train on).

## What is expected from the second MVP?
> Parsing procedure for mp3, wav files
> Simple gRPC server code and proto file to simplify the access from server with pretrained model
> Larger dataset for training: mp3 files, multi-instrument midis, pre-generated accompaniment augmentations of simple tracks
> Model weights v2 uploaded to github

## Which architectures are used/will be used in this project?
> Long Short-Term Memory (LSTM) architecture is used for training and generating.
> Other architectures may be used to classify tracks of the dataset.
> Another LSTM may be used to name tracks (to generate titles for them)
> Transformer models are the possible successors (to use instead of LSTM)

## What are the plans?
> Launch the server which will be able to generate a track upon request given the style and purpose
