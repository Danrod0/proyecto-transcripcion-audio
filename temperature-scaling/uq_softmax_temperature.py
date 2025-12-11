# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:39:47 2024

@author: saul1
"""

#imports
import torch
import numpy as np
from jiwer import wer
import seaborn as sns
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor,WhisperConfig
from datasets import load_dataset, load_metric, Audio
from IPython.display import Audio as AudioDis
import librosa.display
import matplotlib.pyplot as plt
from text_unidecode import unidecode
import torch
from datasets import load_dataset
from datasets import load_dataset, load_metric, Audio
from IPython.display import Audio as AudioDis
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig
#import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor,WhisperConfig
#use CUDA gpu if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import  RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def class UQ_softmax_temperature():
    def temperature_scaled_softmax(self, logits, temperature = 1.0):
    """
    Evaluate the softmax with temperature
    param logits: logits of the model
    param temperature: temperature of the softmax function
    return softmax with temperature
    """
    logits = logits / temperature
    return torch.softmax(logits, dim = -1)


def quantify_certainty_softmax(self, audio_array, model, processor, noise_audios_dataset, sampling_rate = 16000, weight_noise = 0, temperature = 1):
    """
    Quantify the certainty for a specific audio query: The higher the score, the more certain the model is
    param audio_array: query array
    param model: model to evaluate the uncertainty
    param processor: process the input data and transform it to the mel cepstrum based representation
    param noise_audios_dataset: dataset with noise audios, in case we want to contaminate the audio
    param sampling_rate: sampling rate of the audio
    param weight_noise: weight of the noise in the audio
    param temperature: temperature of the softmax function
    return certainty for the query its transcription and its max_probabilities
    """
    if(weight_noise > 0):
      print("Contaminating audio...")
      audio_array = contaminate_audio_array(audio_array, noise_audios_dataset, weight_noise)

    #transform the audio to the input using the required representations
    inputs = processor.feature_extractor(audio_array, return_tensors="pt", sampling_rate = 16_000).input_features
    #output decoder
    forced_decoder_ids = processor.get_decoder_prompt_ids(language = "es", task = "transcribe")


    with torch.no_grad():
        model.train()
        generated_ids = model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            num_return_sequences=1
        )
        #calculate logits of the model, better use logits
        logits = model(inputs, decoder_input_ids=generated_ids).logits
        #calculate softmax...
        probabilities = temperature_scaled_softmax(logits, temperature = temperature)
        #print("probabilities shape ", probabilities.shape)
        #print("probabilities \n", probabilities)
        max_probabilities = torch.max(probabilities, dim = 2).values
        certainty = max_probabilities.mean()
        #print("max_probabilities shape ", max_probabilities.shape)
        #print("max_probabilities \n", max_probabilities)
        #fetch transcriptions in text
        transcriptions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens = False, normalize = True)
        transcription = tildar_oracion(transcriptions[0])
        #print("transcription ", transcription)

    return certainty, transcription, max_probabilities


def quantify_certainty_softmax_dataset(self, dataset_audios, model, processor, noise_audios_dataset, weight_noise = 0, temperature = 1):
  """
  Quantify the score for a given set of audios: The higher the score, the more certain the model is
  param dataset_audios: dataset with audios
  param model: model to evaluate the uncertainty
  param processor: process the input data and transform it to the mel cepstrum based representation
  param noise_audios_dataset: dataset with noise audios, in case we want to contaminate the audio
  return certainty scores for the dataset
  """
  certainties_list = []
  certainties_array_list = []
  transcriptions_list = []
  total_num_audios = len(dataset_audios)
  #noise_audios_dataset = load_dataset("MichielBontenbal/UrbanSounds")
  i = 0
  for audio in dataset_audios:
    print("Transcribing and uq of audio ", i, " of ", total_num_audios)
    i += 1
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    #print("sampling rate ", sampling_rate)
    certainty, transcription, max_probabilities = quantify_certainty_softmax(audio_array, model, processor, noise_audios_dataset, sampling_rate, weight_noise, temperature = temperature)
    certainties_list.append(certainty)
    certainties_array_list.append(max_probabilities)
    transcriptions_list.append(transcription)
  #to tensor...
  certanties_tensor = torch.tensor(certainties_list)
  return certanties_tensor, transcriptions_list, certainties_array_list

    