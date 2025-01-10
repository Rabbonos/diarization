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
            global client # LATER DO SOMETHING ABOUT IT!
            global vad_pipeline #LATER DO SOMETHING ABOUT IT!

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

            #main auido - can be a path or link to audio

            # if file does not exist in local system
            if not os.path.isfile(main_audio):              
                       main_audio =  download_audio(source, main_audio)

            logger.info(main_audio, 'MAIN AUDIO FILE"S PATH')

            #folders / files , sample folders here
            folders_manager = FolderTree(main_folder, 'yandex', participants, ATTEMPTS, ATTEMPTS_INTERVAL, client)

            participant_folder_paths , participants = folders_manager.process_subfolders()

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


#NOT REFACTORED YET ...
import re
from pyannote.audio import Audio
from pyannote.core import Segment


#####################################################################################################

transcribed_audio='''
[0:00:19,000 --> 0:00:21,000] Ермолаева Ирина:  Да ничего страшного Алексей
[0:00:21,000 --> 0:00:23,000] Ермолаева Ирина:  главное что мы все собрались
[0:00:23,000 --> 0:00:25,000] Ермолаева Ирина:  сегодня у нас с вами будет встреча с Александрой
[0:00:25,000 --> 0:00:27,000] Ермолаева Ирина:  Александр у нас
[0:00:27,000 --> 0:00:29,000] Ермолаева Ирина:  Александр как сейчас должность
[0:21:38,000 --> 0:21:41,000] Пчеловодова Александра:  Здесь от вас не ожидается никаких
[0:21:41,000 --> 0:21:44,000] Пчеловодова Александра:  дополнительных описаний, интеграций и так далее.
[0:21:44,000 --> 0:21:48,000] Пчеловодова Александра:  Это чисто такое тезис пользовательской точки зрения.
[0:21:48,000 --> 0:21:51,000] Пчеловодова Александра:  Там вот описание странички, что на ней есть,
[0:21:51,000 --> 0:21:55,000] Пчеловодова Александра:  как те или иные элементы взаимодействуют между собой,
[0:21:55,000 --> 0:21:57,000] Пчеловодова Александра:  как те или иные страницы сметчатся,
[0:21:57,000 --> 0:22:00,000] Пчеловодова Александра:  как те или иные поля, грубо говоря.
[0:22:00,000 --> 0:22:07,900] Пчеловодова Александра:  Соответственно, команда будет проводить параллельно работы по написанию back-end-up для админки,
[0:22:07,900 --> 0:22:13,259] Пчеловодова Александра:  потом фронта и, соответственно, параллельно поддерживать текущие новостные сайты.
[0:22:13,259 --> 0:22:16,799] Пчеловодова Александра:  По ним прилетают порой горящие задачи и так далее.
[0:22:16,799 --> 0:22:21,500] Пчеловодова Александра:  Я хочу сделать также, что специфика работы — это то, что все-таки не дикая, это 24 на 7.
[0:22:21,500 --> 0:22:27,740] Пчеловодова Александра:  То есть если какие-то бывают поломки, и не факт, что там техподдержка может управляться
[0:22:27,740 --> 0:22:31,539] Пчеловодова Александра:  или их обрабатывать, или просто вам нужно как проекту подключиться,
[0:22:31,539 --> 0:22:36,039] Пчеловодова Александра:  то есть могут вам позвонить в любое время дня и ночи, праздник, не праздник, еще что-то.
[0:22:36,039 --> 0:22:39,539] Пчеловодова Александра:  Это специфика сферы, нужно это учитывать.
[0:22:39,539 --> 0:22:45,299] Пчеловодова Александра:  Естественно, такие-то переработки и так далее оплачиваются, но это не та работа,
[0:22:45,299 --> 0:22:49,339] Пчеловодова Александра:  где ты закончил работать в 7 вечера, закрыл ноутбук и ушел.
'''

pattern_time_start = re.compile('(?<=\[).*(?=-->)')
pattern_time_end = re.compile('(?<=-->).*(?=])')
pattern_name = re.compile(r'(?<=\])\s*(.*?)(?=:)\s*')
pattern_text = re.compile('(?<=:) .*')
transcribed_audio_lines= transcribed_audio.split('\n')
transcribed_audio_lines=[line for line in transcribed_audio_lines if len(line)>3]
print('всего строк: ',len(transcribed_audio_lines))
print(transcribed_audio_lines)
count=0
def str_to_seconds(time):
        time = time.split(':')
        hours = int(time[0])  # Convert hours to integer
        minutes = int(time[1])  # Convert minutes to integer
        seconds = float(time[2].replace(',', '.'))  # Convert seconds to float, replacing ',' with '.'
        return  hours * 3600 + minutes * 60 + seconds

for line in transcribed_audio_lines:

    last_number=0
    flag=False
    name_= re.search ( pattern_name, line).group(0).strip()
    text= re.search ( pattern_text, line).group(0).strip()
    time_start=re.search ( pattern_time_start, line).group(0).strip() #need to convert from '0:01:49,000' to seconds
    time_end= re.search ( pattern_time_end, line).group(0).strip() #need to convert from '0:01:49,000' to seconds
    time_start = str_to_seconds(time_start)
    time_end= str_to_seconds(time_end)
    
    folders_manager = FolderTree(main_folder, 'yandex', participants, ATTEMPTS, ATTEMPTS_INTERVAL, client)

    participant_folder_paths , participants = folders_manager.process_subfolders()

    if not name_ in participants:
        participants.append(name_)
        participant_folder_paths.append(os.path.join(main_folder, name_))

    sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, mode, client, ATTEMPTS, ATTEMPTS_INTERVAL)

    if name_ and flag:

            i=last_number
            path= main_audio.split('/')[-1].split('.')[0]
            path= f'{path}-{name_}_{i+1}.wav'
            sample_path=os.path.join(sample_path,  path )
            sample_audio = Audio()
            sample_clip = Segment( time_start , time_end)

            sample_waveform, _ = sample_audio.crop('audio.wav', sample_clip)

            sample_waveform=sample_waveform.squeeze(0)
            sample_waveform=sample_waveform.squeeze(0)

            # sample_waveform to audio
            sf.write('cropped_fragment.wav', sample_waveform, samplerate=16000, format='WAV')

            result_of_vad= vad_pipeline('cropped_fragment.wav')
            time_segments=result_of_vad.get_timeline() # time segments of active speech

            if len(time_segments)>0:
                            main_wave_audio = Audio()
                            waveforms=[]
                            duration= get_duration('cropped_fragment.wav')

                            for time_segment in time_segments:
                                  speech_start = time_segment.start
                                  speech_end = time_segment.end
                                  speech_end = min(duration, speech_end)
                                  main_clip = Segment( speech_start , speech_end)
                                  main_waveform, _ = main_wave_audio.crop('cropped_fragment.wav', main_clip)
                                  main_waveform=main_waveform.squeeze(0)
                                  main_waveform=main_waveform.squeeze(0)
                                  waveforms.append(main_waveform)
                            if waveforms:
                                        final_waveform = np.concatenate(waveforms, axis=-1)
                            # final_waveform to audio
                            if mode=='yandex':
                                    sf.write('sample_sample.wav', final_waveform, samplerate=16000, format='WAV')
                                    client.upload("sample_sample.wav", sample_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)

                            else: sf.write(sample_path, final_waveform, samplerate=16000, format='WAV')

            else:
                  if mode=='yandex':
                                sf.write('sample_sample.wav', sample_waveform, samplerate=16000, format='WAV')

                                client.upload("sample_sample.wav", sample_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
                  else: sf.write(sample_path, sample_waveform, samplerate=16000, format='WAV')

            count+=1
            print('сохраняет в :', sample_path)

print('Фрагментов добавленно: ',count )





###########################################################################



# Обновление векторов (только тех участников, которых отметили в настройках + у которых было обнаружено аудио т.к. без аудио не будет векторов).

#ADD LOGIC THAT CREATES FOLDERS IF THEY ARE NOT PRESENT !
def handle_yandex_json(client, json_path, folder_name, name, dict_for_json):
    """
    Handles JSON operations with Yandex disk, ensuring complete file transfers
    """
    try:
        # Create a temporary local file for operations
        temp_local_path = f"temp_{name}.json"

        # Download existing JSON if it exists
        try:

            client.download(json_path, temp_local_path, n_retries = attempts  , retry_interval =attempts_interval)
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


        client.remove(json_path, permanently=True, n_retries = attempts  , retry_interval =attempts_interval)


        # Upload new file and verify
        client.upload(temp_local_path, json_path, n_retries = attempts  , retry_interval =attempts_interval)

        # Verify upload was successful
        client.download(json_path, f"verify_{name}.json", n_retries = attempts  , retry_interval =attempts_interval)
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

# Step 1: List sample folders
if use_yandex:

                sample_folders_names = client.listdir(os.path.join(main_folder, 'Образцы голоса'), n_retries = attempts  , retry_interval =attempts_interval)


                sample_folders_names= [x['path'] for x in sample_folders_names]

else: sample_folders_names = [ os.path.join(main_folder, 'Образцы голоса',x)  for x  in os.listdir(os.path.join(main_folder, 'Образцы голоса')) ]

final_list = []
temp_list_folder = []
# Step 2: Create a list of words and corresponding folder names for sample folders
for sample_folder in sample_folders_names:
                sample_folder_words = normalize_string(sample_folder.strip()).split('/')[-1].strip().split(' ') # Split folder name into words
                temp_list_folder.append([sample_folder_words, sample_folder])  # Store words and name

tem_list_participant = []
# Step 3: Create a list of words and corresponding participant names
for participant in participants_restore:
    participant_words = normalize_string(participant.strip()).split(' ')
    tem_list_participant.append([participant_words, participant])  # Store words and name

# Step 4: Compare the lists and find matches
double_list=[]
for y in tem_list_participant:
        flag=False
        for x in temp_list_folder:
            # Check if any two values in x[0] match any two values in y[0]
            if len(set(x[0]).intersection(set(y[0]))) >= 2:  # At least 2 matching words
                final_list.append(x[1])  # Append the folder name to the final list
                double_list.append([x[1],y[0]])
                flag=True
                break
        if flag==False:
            names_string=''
            for string_ in y[0]:
                names_string =names_string +' '+ string_
            names_string=names_string.strip()
            folder= os.path.join (main_folder, 'Образцы голоса', names_string)
            print(f'для участника {y[0]} не нашлась папка, создаю новую:{ folder }')

            client.mkdir(folder, n_retries = attempts  , retry_interval =attempts_interval)

            final_list.append(folder)
            double_list.append([folder,y[0]])

# Update sample_folders with final_list
sample_folders_restore = [x[0] for x in double_list]
participants = [' '.join(x[1]) for x in double_list]

for i in range (attempts):

      try:

              if use_yandex:
                            with client:

                                        sample_vectors = []
                                        for folder,name in zip(sample_folders_restore, participants):

                                                            found_json = False

                                                            ###########

                                                            files= client.listdir(folder, n_retries = attempts  , retry_interval =attempts_interval)
                                                            for file in files:
                                                                      file=file['path']
                                                                      if file.endswith('.json'):
                                                                              sample_vectors.append([file, name, folder])
                                                                              found_json = True

                                                            if not found_json:
                                                                    empty_json_path = f"{name}.json"
                                                                    empty_json_path_for_upload = os.path.join(folder, f"{name}.json")
                                                                    with open(empty_json_path, 'w') as f:
                                                                              json.dump({}, f)  # Creates an empty JSON file with an empty dictionary

                                                                              client.upload(empty_json_path, empty_json_path_for_upload , n_retries = attempts  , retry_interval =attempts_interval)
                                                                    sample_vectors.append([empty_json_path, name, folder])
              else:
                    sample_vectors = []
                    for folder,name in zip(sample_folders_restore, participants):

                                        found_json = False
                                        for file in os.listdir(folder):
                                                  if file.endswith('.json'):
                                                          sample_vectors.append([file, name, folder])
                                                          found_json = True

                                        if not found_json:
                                                empty_json_path = os.path.join(folder, f"{name}.json")
                                                with open(empty_json_path, 'w') as f:
                                                          json.dump({}, f)  # Creates an empty JSON file with an empty dictionary
                                                sample_vectors.append([empty_json_path, name, folder])

              if use_yandex:
                  for jsonfile, name, folder_name in sample_vectors:
                    try:
                        ############


                        client.remove(jsonfile, permanently=True)

                    except:pass
              else:
                  for jsonfile, name, folder_name in sample_vectors:
                    try:
                        os.remove(os.path.join(folder_name,jsonfile))
                    except:
                        pass


              if use_yandex:
                            with client:

                                        sample_vectors = []
                                        for folder,name in zip(sample_folders_restore, participants):
                                                            found_json = False

                                                            ##############

                                                            files= client.listdir(folder, n_retries = attempts  , retry_interval =attempts_interval)
                                                            for file in files:

                                                                      file=file['path']
                                                                      if file.endswith('.json'):
                                                                              sample_vectors.append([file, name, folder])
                                                                              found_json = True

                                                            if not found_json:
                                                                    temp_dict={}
                                                                    empty_json_path = f"{name}.json"
                                                                    empty_json_path_for_upload = os.path.join(folder, f"{name}.json")
                                                                    with open(empty_json_path, 'w') as f:
                                                                              json.dump(temp_dict, f)  # Creates an empty JSON file with an empty dictionary
                                                                              ########

                                                                              client.upload(empty_json_path, empty_json_path_for_upload, n_retries = attempts  , retry_interval =attempts_interval )
                                                                    sample_vectors.append([empty_json_path, name, folder])


                                        sample_audios = []
                                        for folder,name in zip(sample_folders_restore, participants):
                                                  ##########

                                                  files= client.listdir(folder, n_retries = attempts  , retry_interval =attempts_interval)
                                                  for file in files:
                                                      file_type=file['type']
                                                      file=file['path']
                                                      if not file.endswith('.json') and file_type!='dir':
                                                          sample_audios.append([file, name, folder])
              else:
                    sample_vectors = []
                    for folder,name in zip(sample_folders_restore, participants):

                                        found_json = False
                                        for file in os.listdir(folder):
                                                  if file.endswith('.json'):
                                                          sample_vectors.append([file, name, folder])
                                                          found_json = True

                                        if not found_json:
                                                temp_dict={}
                                                empty_json_path = os.path.join(folder, f"{name}.json")
                                                with open(empty_json_path, 'w') as f:
                                                          json.dump(temp_dict, f)  # Creates an empty JSON file with an empty dictionary
                                                sample_vectors.append([empty_json_path, name, folder])


                    sample_audios = []
                    for folder,name in zip(sample_folders_restore, participants):

                              for file in os.listdir(folder):
                                  if not file.endswith('.json')and os.path.isfile(os.path.join(folder,file)):
                                      sample_audios.append([file, name, folder])


              print(sample_audios,'sample_audios')
              for file, name, folder_name in sample_audios:

                        if use_yandex:
                            sample_path=file

                        else:
                            sample_path = os.path.join( main_folder ,'Образцы голоса', folder_name ,file)

                        if use_yandex:
                            #######

                            client.download(sample_path, 'sample_path', n_retries = attempts  , retry_interval =attempts_interval )
                        dict_for_json={}
                        sample_audio = Audio()
                        sample_wav_path = f'sample_voice.wav'

                        if use_yandex:
                                      subprocess.call(['ffmpeg', '-i', 'sample_path', '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-frame_size', '400', '-y', sample_wav_path])
                        else:
                                      subprocess.call(['ffmpeg', '-i', sample_path, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-frame_size', '400', '-y', sample_wav_path]) #некоторые библиотеки так требуют

                        with contextlib.closing(wave.open(sample_wav_path, 'r')) as f:
                            sample_frames = f.getnframes()
                            sample_rate = f.getframerate()
                            sample_duration = sample_frames / float(sample_rate)

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

                                      if use_yandex:
                                            json_path=os.path.join(folder_name ,  f'{name}.json')
                                      else:
                                            json_path= os.path.join(  main_folder ,'Образцы голоса', folder_name ,  f'{name}.json')

                                      found=False
                                      if use_yandex:
                                              ############

                                              files= client.listdir(folder_name, n_retries = attempts  , retry_interval =attempts_interval)
                                              for path in files:
                                                path=path['path']
                                                if normalize_string(path) == normalize_string(json_path):
                                                    found=True
                                                    break
                                              if found==True:
                                                    ############
                                                    handle_yandex_json(client=client,
                                                                      json_path=json_path,
                                                                      folder_name=folder_name,
                                                                      name=name,
                                                                      dict_for_json=dict_for_json)

                                      else:
                                            for path in os.listdir( os.path.join(  main_folder ,'Образцы голоса', folder_name )):
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


              break
      except Exception as e:
              sleep(attempts_interval)
              print(f'ошибка {str(e)}')
