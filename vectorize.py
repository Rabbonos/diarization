from pyannote.audio import Audio
from pyannote.core import Segment
import os
import json
from file_manager import FolderTree, fetch_vectors_and_audio_files, normalize_string
import warnings
from video import get_duration
from audio_processing import read_json, standardize_audio
from typing import Callable,List
# Обновление векторов (только тех участников, которых отметили в настройках + у которых было обнаружено аудио т.к. без аудио не будет векторов).

#ADD LOGIC THAT CREATES FOLDERS IF THEY ARE NOT PRESENT !
def handle_yandex_json(client, json_path, folder_name, name, dict_for_json,  ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int=3):
    """
    Handles JSON operations with Yandex disk, ensuring complete file transfers
    """
    try:
        # Create a temporary local file for operations
        temp_local_path = f"temp_{name}.json"
        # Download existing JSON if it exists
        try:
            client.download(json_path, temp_local_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
            with open(temp_local_path, 'r') as f:
                existing_data = json.load(f)
        except:
            existing_data = {}

        # Update with new data
        existing_data.update(dict_for_json)

        # Write updated data to temporary file
        with open(temp_local_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())  # Ensure all data is written to disk

        # Remove existing file on Yandex if it exists
        client.remove(json_path, permanently=True, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)

        # Upload new file and verify
        client.upload(temp_local_path, json_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)

        # Verify upload was successful
        client.download(json_path, f"verify_{name}.json", n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
        with open(f"verify_{name}.json", 'r') as f:
            verify_data = json.load(f)

        if verify_data != existing_data:
            raise Exception("Upload verification failed - data mismatch")

        # Cleanup temporary files
        os.remove(temp_local_path)
        os.remove(f"verify_{name}.json")

        return True

    except Exception as e:
        print(f"Error handling JSON for {name}: {str(e)}")
        return False
    

def clear_vectors(participants:List, sample_vectors:List[str,str,str], client:Callable=None):
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
def upgrade_vectors(main_folder:str, participants:List[str], embedding_model:str, embedding_model_name:str, embedding_models_group1:List , speaker_model=None, client:Callable=None, ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int=3):
        '''Upgrades vector json files for specified participants
        
            Args:


            Returns:
                None
        '''
        folders_manager = FolderTree(main_folder, 'yandex', participants, ATTEMPTS, ATTEMPTS_INTERVAL, client)

        #sample_folders_restore participants
        participant_folder_paths , participants = folders_manager.process_subfolders()

        sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, client, ATTEMPTS, ATTEMPTS_INTERVAL)

        sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants,client, ATTEMPTS, ATTEMPTS_INTERVAL)

        print(sample_audios,'sample_audios')
        for sample_path, name, folder_name in sample_audios:

                if client:
                    client.download(sample_path, 'sample_path', n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL )
                
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

                try:
                    sample_clip = Segment(0, sample_duration)
                    sample_waveform, _ = sample_audio.crop(sample_wav_path, sample_clip)
                    if embedding_model_name in embedding_models_group1:
                        sample_embedding= embedding_model(sample_waveform[None])
                    else:
                        sample_embedding=embedding_model(speaker_model, sample_waveform[None]).cpu()

                    dict_for_json[f'{sample_path}']=[sample_embedding.tolist()[0]]
                    #
                    for json_path, name_ , folder_name_ in sample_vectors:
                        if normalize_string(name) == normalize_string(name_):
                                if client:
                                    json_path=os.path.join(folder_name ,  f'{name}.json')
                                else:
                                    json_path= os.path.join(  main_folder ,'Образцы голоса', folder_name ,  f'{name}.json')
                                found=False
                                if client:
                                        files= client.listdir(folder_name, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
                                        for path in files:
                                            path = path['path']
                                            if normalize_string(path) == normalize_string(json_path):
                                                found=True
                                                break
                                        if found==True:
                                            handle_yandex_json(client=client,
                                                                json_path=json_path,
                                                                folder_name=folder_name,
                                                                name=name,
                                                                dict_for_json=dict_for_json)
                                else:
                                    for path in os.listdir( os.path.join( main_folder ,'Образцы голоса', folder_name )):
                                            path= os.path.join(  main_folder ,'Образцы голоса', folder_name , path )
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


