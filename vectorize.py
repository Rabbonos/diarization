from pyannote.audio import Audio
from pyannote.core import Segment
import os
import json
from file_manager import FolderTree, fetch_vectors_and_audio_files, normalize_string
import warnings
from video import get_duration
from audio_processing import read_json, standardize_audio
from typing import Callable,List
from yandex import handle_yandex_json
from setttings import Settings
import yadisk
from audio_processing import AudioProcessor
# Обновление векторов (только тех участников, которых отметили в настройках + у которых было обнаружено аудио т.к. без аудио не будет векторов).

    

def clear_vectors(participants:List, sample_vectors:List, client:Callable=None):
        '''clearing vectors of some participants we want to upgrade before upgrading them''' 
        if client:
            for jsonfile, name, folder_name in sample_vectors:
                if name in participants:
                    try:
                        client.remove(jsonfile, permanently=True)
                    except:
                        warnings.warn(f'For {name}, jsonfile {jsonfile} could not be deleted')
        else:
            for jsonfile, name, folder_name in sample_vectors:
                if name in participants:
                    try:
                        os.remove(os.path.join(folder_name,jsonfile))
                    except:
                        warnings.warn(f'For {name}, jsonfile {jsonfile} could not be deleted')

#participants only the ones we specify (new setting for that)
#main_folder:str, participants:List[str], embedding_model:str, embedding_model_name:str, embedding_models_group1:List , speaker_model=None, client:Callable=None, ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int=3
def upgrade_vectors(settings:Settings):
        '''Upgrades vector json files for specified participants
        
            Args:


            Returns:
                None
        '''
        #anyway all is cached , so no need to give from transcibe model here
        audio_models_loading = AudioProcessor(settings)
        embedding_model, speaker_model = audio_models_loading.embedding_model, audio_models_loading.speaker_model 

        if settings.mode=='yandex':
                client = yadisk.Client(token=settings.token_yandex  )
        else:
             client = None
        folders_manager = FolderTree(settings.main_folder, settings.PARTICIPANTS, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL, client)

        #sample_folders_restore participants
        participant_folder_paths , participants = folders_manager.process_subfolders()

        sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, client,  settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL)

        #sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants,client,  settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL)

        print(sample_audios,'sample_audios')
        for sample_path, name, folder_name in sample_audios:

                if client:
                    client.download(sample_path, 'sample_path', n_retries = settings.ATTEMPTS  , retry_interval =settings.ATTEMPTS_INTERVAL )
                
                dict_for_json={}
                sample_audio = Audio()
                sample_wav_path = f'sample_voice.wav'
                
                if client:
                        standardize_audio('sample_path', sample_wav_path)
                else:
                        standardize_audio(sample_path, sample_wav_path)

                sample_duration = get_duration(sample_wav_path)

                if sample_duration<1:
                    continue
                ################################take embedding model from transcribe or define  a new one here + speaker model
                try:
                    sample_clip = Segment(0, sample_duration)
                    sample_waveform, _ = sample_audio.crop(sample_wav_path, sample_clip)
                    if settings.embedding_model_name in settings.embedding_models_group1:
                        sample_embedding= embedding_model(sample_waveform[None])
                    else:
                        sample_embedding=embedding_model(speaker_model, sample_waveform[None]).cpu()

                    dict_for_json[f'{sample_path}']=[sample_embedding.tolist()[0]]
                    #
                    for json_path, name_ , folder_name_ in sample_vectors:
                        if normalize_string(name) == normalize_string(name_):
                                if client:
                                    json_path=os.path.join(folder_name ,  f'{name}.json')
                                    json_path = json_path.replace('\\','/')
                                else:
                                    json_path= os.path.join(  settings.main_folder ,'Образцы голоса', folder_name ,  f'{name}.json')
                                found=False
                                if client:
                                        files= client.listdir(folder_name, n_retries = settings.ATTEMPTS  , retry_interval =settings.ATTEMPTS_INTERVAL)
                                        for path in files:
                                            path = path['path']
                                            if normalize_string(path) == normalize_string(json_path):
                                                found=True
                                                break
                                        if found==True:
                                            handle_yandex_json(client=client,
                                                                json_path=json_path,
                                                                name=name,
                                                                dict_for_json=dict_for_json)
                                else:
                                    for path in os.listdir( os.path.join( settings.main_folder ,'Образцы голоса', folder_name )):
                                            path= os.path.join(  settings.main_folder ,'Образцы голоса', folder_name , path )
                                            if normalize_string(path.strip()) == normalize_string(json_path.strip()):
                                                found=True
                                                break
                                    if found==True:
                                            vectorized_sample_old =read_json(json_path)
                                            vectorized_sample_old.update(dict_for_json)
                                            with open(json_path, 'w') as f:
                                                    json.dump(vectorized_sample_old, f, indent=4)
                                            print(name, json_path , sample_path)
                except Exception as e:
                    print(f'ошибка (иногда бывает если аудио слишком короткое - нормально это, если не короткое то реально ошибка): {str(e)}')


