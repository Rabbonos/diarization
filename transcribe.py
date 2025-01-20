from speechbrain.inference.separation import SepformerSeparation
import numpy as np
import os
import warnings
from sklearn.metrics import pairwise_distances
import yadisk 
from audio_processing import fragment, download_audio, preprocess_segments, main_audio_preprocessing
from audio_processing import AudioProcessor, get_sample_embeddings , replace_overlaps, use_vectors , get_embeddings_main_audio
from clustering import ClusteringModel, assign_speakers, assign_speakers_individually
from file_manager import FolderTree, fetch_vectors_and_audio_files
from video import get_duration
from manage_output import print_transcription, save_transcription, write_to_word, remove_junk
from Logging_config import configure_logging
import logging
from yandex import get_new_ytoken
from constants import *
from setttings import Settings

configure_logging()
logger = logging.getLogger('newlogger')


def start(settings: Settings):
            '''the function that starts transcription'''

            main_audio_restore = settings.main_audio
            speaker_model=None 
            main_audio = settings.main_audio
            #junk_list = []
            #junk_list.extend([TEMP_FOLDER_MAIN, TEMP_FOLDER_FRAGMENTS, CLIPPED_SEGMENTS]) #for now only this

            #используем яндекс диск
            if settings.mode=='yandex':
                client = yadisk.Client(token=settings.token_yandex  )
            else:
                 client=None

            if settings.get_token =='да':
                   get_new_ytoken()

            #main auido - can be a path or link to audio
            if not os.path.isfile(main_audio):              
                       main_audio =  download_audio(settings.source, main_audio)

            logger.info(main_audio, 'MAIN AUDIO FILE"S PATH')

            #folders / files , sample folders here
            folders_manager = FolderTree(settings.main_folder, settings.PARTICIPANTS, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL, client)

            participant_folder_paths , participants = folders_manager.process_subfolders()

            # os.makedirs('content',exist_ok=True) #CORRECT IT !

            ###FRAGEMENTS HERE
            fragments = fragment(settings.interval, main_audio , settings.divide_interval, settings.TEMP_FOLDER_MAIN, settings.TEMP_FOLDER_FRAGMENTS )
            logger.info(participant_folder_paths, 'participant_folder_paths')

            #HERE VECTORS AND AUDIO DATA
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, client, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL)
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

            #we lose here !
            logger.info('participants_with_samples', participants_with_samples, folders_with_samples, ' folders_with_samples')
            #zdes u nas iz sample_vectors, sample_audios est participants bez audio
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, client, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL)
            
            logger.info('POSLE UDALENIYA BEZ AUDIO PARTICIPANTS:',sample_vectors, 'sample_vectors', sample_audios, 'sample_audios')
            # A LOT OF AUDIPROCESSING PREPARATIONS HERE
            audio_models_loading = AudioProcessor(settings)
            embedding_model= audio_models_loading.embedding_model
            embedding_model_dimension = audio_models_loading.embedding_model_dimension
            model= audio_models_loading.model
            vad_pipeline=None 
            pipeline=None

            if settings.Silence:         
                  vad_pipeline= audio_models_loading.vad_pipeline

            if settings.Accuracy_boost:
                  pipeline= audio_models_loading.diarization_pipeline

            if settings.save_txt:
                  main_audio_file_name = main_audio_restore.split('/')[-1].split('.')[0]
                  file_path = f"Transcription_{main_audio_file_name}.txt"
            else:
                  file_path=None
 
            #main_audio loading 
            
            result = main_audio_preprocessing(model,settings.main_audio_wav_path,settings.is_fragment, main_audio, fragments, settings.Silence, settings.Accuracy_boost, vad_pipeline, pipeline)
            duration = get_duration(settings.main_audio_wav_path)
            segments = result["segments"]
            sep_model = SepformerSeparation.from_hparams( "speechbrain/sepformer-whamr16k" )
            overlap_timestamps = audio_models_loading.remove_overlap(settings.main_audio_wav_path)

            # deleting old segments and replacing them with new segments HERE
            segments = replace_overlaps(segments, sep_model, model, overlap_timestamps)
            #sorting because I potentially changed the order in the previous lines
            segments = sorted(segments, key=lambda segment: segment['start'])
            segments = preprocess_segments(segments)

            logger.info(segments, 'SEGMENTS')
            #GETTING SAMPLE VECTORS OR USING PRE-MADE SAMPLE VECTORS
            if settings.Vector:
                    sample_embeddings = use_vectors(settings, sample_vectors, client)
            else:
                    if settings.Voice_sample_exists:
                                sample_embeddings = get_sample_embeddings(settings,sample_audios, sample_vectors, embedding_model, speaker_model,client)

            #GETTING EMBEDDINGS OF MAIN AUDIO
            embeddings = get_embeddings_main_audio(settings,settings.main_audio_wav_path, segments, embedding_model, embedding_model_dimension, speaker_model)
            embeddings = np.nan_to_num(embeddings)
            embedding_size = embeddings.shape[1]
            small_vector = np.full(embedding_size, -1e10)  # or np.zeros(embedding_size)
            # Replace rows where embeddings are all zeros with small_vector
            embeddings[~np.any(embeddings, axis=1)] = small_vector

            distances = pairwise_distances(embeddings, metric='cosine')  # or use any other metric
            distances = distances[np.triu_indices_from(distances, k=1)]

            if settings.cluster_settings.distance_threshold==None:
                    settings.cluster_settings.distance_threshold = np.mean(distances) +  np.std(distances)/2
                
            #CLUSTERING BLOCK
            clustering_manager = ClusteringModel(settings.cluster_settings)

            if settings.Voice_sample_exists:
                if settings.clustering:
                    assigned_speakers = assign_speakers(clustering_manager, embeddings, segments, sample_embeddings, settings.SIMILARITY_THRESHOLD)
                    
            if not settings.clustering:
                assigned_speakers = assign_speakers_individually(segments, embeddings, sample_embeddings)

            #MAKE IT DIFFERENT, READ 'TO ADD'
            if not settings.Voice_sample_exists:
                cluster_labels= clustering_manager.cluster(embeddings) ###
                assigned_speakers = ["Участник (не определён)" for i in cluster_labels]

            #OUTPUT BLOCK
            if settings.save_mode == 'text':
                    save_transcription(file_path, segments, assigned_speakers, duration, settings.metki)

            elif settings.save_mode == 'word':
                    write_to_word(segments,assigned_speakers , duration, settings.metki, 'transcription.docx')

            print_transcription(segments, assigned_speakers, duration, settings.metki)


            #junk object should be in constant and used in all files
            # for junk in junk_list:
            #         remove_junk(junk)
            return vad_pipeline, client
if __name__ == '__main__':
      #start()
      pass