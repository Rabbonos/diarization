#https://github.com/zedr/clean-code-python

# # !pip install nemo_toolkit[all] (при небходимости раздокументировать)
# !pip install onnxruntime
# !pip install pyannote-whisper
# !pip install yadisk[sync_defaults]
# !pip install -q gdown
# !pip install setuptools==59.5.0
# !pip install speechbrain==0.5.16
# !git clone https://github.com/yinruiqing/pyannote-whisper.git
# !cd pyannote-whisper && pip install .

#use rclone to mount google drive?


#TO ADD: IF only someone did not have sample, only him to denote as 'undefinde', not all 
#TO ADD: tones 
#TO ADD: TRAINING 
#TO ADD: TRAINING DATA PREPARATION
#TO ADD: async stuff + caching



from speechbrain.inference.separation import SepformerSeparation
import numpy as np
import os
import warnings
#from google.colab import files
from sklearn.metrics import pairwise_distances
#from google.colab import userdata
# import nemo.collections.asr as nemo_asr (раздокументировать при необходимости)
import soundfile as sf
import onnxruntime
import json
import yadisk #yandex

from audio_processing import fragment, download_audio, preprocess_segments, main_audio_preprocessing, AudioProcessor, get_sample_embeddings , replace_overlaps, use_vectors , get_embeddings_main_audio
from clustering import ClusteringModel, assign_speakers, assign_speakers_individually
from file_manager import FolderTree, fetch_vectors_and_audio_files
from constants import *
from video import get_duration
from manage_output import print_transcription, save_transcription, write_to_word, remove_junk
from Logging_config import configure_logging
import logging


configure_logging()
logger = logging.getLogger('newlogger')

def get_new_ytoken():
        
        '''get new yandex token'''

        with yadisk.Client(client_id, client_secret) as client:
            url = client.get_code_url()

            print(f"Go to the following url: {url}")
            code = input("Enter the confirmation code: ")

            try:
                response = client.get_token(code)
            except yadisk.exceptions.BadRequestError:
                print("Bad code")
                return
            
            client.token = response.access_token
            token_yandex=client.token
            print(client.token)
            if client.check_token():

                print("Sucessfully received token!")
            else:
                print("Something went wrong. Not sure how though...")


def start():
            participants_restore = participants
            use_yandex=False
            main_audio_restore= main_audio
            speaker_model=None 
            junk = []
            junk.append(TEMP_FOLDER_MAIN, TEMP_FOLDER_FRAGMENTS, CLIPPED_SEGMENTS) #for now only this

            #используем яндекс диск
            if mode=='yandex':
                client = yadisk.Client(token=token_yandex  )
            else:
                 client=None

            if get_token =='да':
                   get_new_ytoken()

            #главный файл (путь)
            #main auido - ssilka / abs. path k audio
            # if file does not exist in local system
            if not os.path.isfile(main_audio):              
                       main_audio =  download_audio(source, main_audio)

            logger.info(main_audio, 'MAIN AUDIO FILE"S PATH')
            #folders / files , sample folders here
            folders_manager = FolderTree(main_folder, 'yandex', ATTEMPTS, ATTEMPTS_INTERVAL, client)

            participant_folder_paths , participants = folders_manager.process_subfolders()
            #participants restore
            #folders restore

            ###FRAGEMENTS HERE
            fragments = fragment(interval, main_audio , divide_interval, TEMP_FOLDER_MAIN, TEMP_FOLDER_FRAGMENTS )
            
            logger.info(participant_folder_paths, 'participant_folder_paths')
            #HERE VECTORS AND AUDIO DATA
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, mode, client, ATTEMPTS, ATTEMPTS_INTERVAL)

            logger.info(sample_vectors, 'sample_vectors', sample_audios, 'sample_audios')
            # Remove participants who don't have any audio samples
            participants_with_samples = [name for name in participants if any(name == name_ for _, name_, _ in sample_audios)]

            # Create a new list of folders corresponding to participants with samples
            folders_with_samples = [folder for folder, name in zip(participant_folder_paths, participants) if name in participants_with_samples]

            if len(participants_with_samples)!=len(participants):
                warnings.warn(f'Some participants did not have audio samples, listing those that had \
                        {participants_with_samples}')

            # Update the participants and folders lists
            participants = participants_with_samples
            folders = folders_with_samples

            logger.info('participants_with_samples', participants_with_samples, folders_with_samples, ' folders_with_samples')
            #zdes u nas iz sample_vectors, sample_audios est participants bez audio
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, mode, client, ATTEMPTS, ATTEMPTS_INTERVAL)
            
            logger.info('POSLE UDALENIYA BEZ AUDIO PARTICIPANTS:',sample_vectors, 'sample_vectors', sample_audios, 'sample_audios')
            # A LOT OF AUDIPROCESSING PREPARATIONS HERE
            audio_models_loading = AudioProcessor(language, modeltype, embedding_model_name, HUGGINGFACE_TOKEN, Accuracy_boost, Silence )
            embedding_model= AudioProcessor.embedding_model
            embedding_model_dimension = AudioProcessor.embedding_model_dimension
            model= AudioProcessor.model
            vad_pipeline=None 
            pipeline=None
            embedding_models_group1 = audio_models_loading.embedding_models_group1

            if Silence:         
                  vad_pipeline= AudioProcessor.vad_pipeline

            if Accuracy_boost:
                  pipeline= AudioProcessor.diarization_pipeline

            if save_txt:
                  file_path = f"Определение голосов расшифровки Whisper - {main_audio_restore}.txt"
            else:
                  file_path=None
 
            #main_audio loading 
            result = main_audio_preprocessing(model,'audio.wav',is_fragment, main_audio, fragments, Silence, Accuracy_boost, vad_pipeline, pipeline)
            duration = get_duration('audio.wav')
            segments = result["segments"]
            sep_model = SepformerSeparation.from_hparams( "speechbrain/sepformer-whamr16k" )
            overlap_timestamps = AudioProcessor.remove_overlap()

            # deleting old segments and replacing them with new segments HERE
            segments = replace_overlaps(segments, sep_model, model, overlap_timestamps)
            #sorting because I potentially changed the order in the previous lines
            segments = sorted(segments, key=lambda segment: segment['start'])
            segments = preprocess_segments(segments)

            logger.info(segments, 'SEGMENTS')
            #GETTING SAMPLE VECTORS OR USING PRE-MADE SAMPLE VECTORS
            if Vector:
                    sample_embeddings = use_vectors(sample_vectors, client)
            else:
                    if Voice_sample_exists:
                                sample_embeddings = get_sample_embeddings(sample_audios, sample_vectors, embedding_model, speaker_model, embedding_models_group1, client)

            #GETTING EMBEDDINGS OF MAIN AUDIO
            embeddings = get_embeddings_main_audio('audio.wav', segments, embedding_model, embedding_model_dimension, embedding_models_group1, speaker_model)
            embeddings = np.nan_to_num(embeddings)
            embedding_size = embeddings.shape[1]
            small_vector = np.full(embedding_size, -1e10)  # or np.zeros(embedding_size)
            # Replace rows where embeddings are all zeros with small_vector
            embeddings[~np.any(embeddings, axis=1)] = small_vector


            distances = pairwise_distances(embeddings, metric='cosine')  # or use any other metric
            distances = distances[np.triu_indices_from(distances, k=1)]

            if DISTANCE_THRESHOLD==None:
                    DISTANCE_THRESHOLD = np.mean(distances) +  np.std(distances)/2

            #CLUSTERING BLOCK
            clustering_manager = ClusteringModel(settings)

            if Voice_sample_exists:
                if clustering:
                    assigned_speakers = assign_speakers(clustering_manager, embeddings, segments, sample_embeddings, SIMILARITY_THRESHOLD)
                    
            if not clustering:
                assigned_speakers = assign_speakers_individually(segments, embeddings, sample_embeddings)

            #MAKE IT DIFFERENT, READ 'TO ADD'
            if not Voice_sample_exists:
                cluster_labels= clustering_manager.cluster(embeddings) ###
                assigned_speakers = ["Участник (не определён)" for i in cluster_labels]

            #OUTPUT BLOCK
            if save_mode == 'text':
                    save_transcription(file_path, segments, assigned_speakers, duration, metki)
            elif save_mode == 'word':
                    write_to_word(segments,assigned_speakers , duration, metki, 'transcription.docx')
            else:
                    print_transcription(segments, assigned_speakers, duration, metki)

            if file_path!=None:
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        transcript = f.read()
                    print(transcript)
                except Exception as e:
                    print(f'For unknown reasons transcription could not be printed, {str(e)}')

            remove_junk(junk)
              
start()