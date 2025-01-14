import re
from file_manager import FolderTree
import os
from typing import Callable
from file_manager import fetch_vectors_and_audio_files
import soundfile as sf
from pyannote.audio import Audio
from pyannote.core import Segment
import numpy as np
#00:00:00-00:01:00

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


def str_to_seconds(time):
        time = time.split(':')
        hours = int(time[0])  # Convert hours to integer
        minutes = int(time[1])  # Convert minutes to integer
        seconds = float(time[2].replace(',', '.'))  # Convert seconds to float, replacing ',' with '.'
        return  hours * 3600 + minutes * 60 + seconds


def parse_text(line:str):
        '''Extracts time, name, .. from a string (line)'''        

        pattern_time_start = re.compile('(?<=\[).*(?=-->)')
        pattern_time_end = re.compile('(?<=-->).*(?=])')
        pattern_name = re.compile(r'(?<=\])\s*(.*?)(?=:)\s*')
        pattern_text = re.compile('(?<=:) .*')

        name= re.search ( pattern_name, line).group(0).strip()
        time_start=re.search ( pattern_time_start, line).group(0).strip() #need to convert from '0:01:49,000' to seconds
        time_end= re.search ( pattern_time_end, line).group(0).strip() #need to convert from '0:01:49,000' to seconds
        text= re.search ( pattern_text, line).group(0).strip()

        return time_start, time_end , name, text

def save_sample(sample_path:str, waveform:'np.array'= None, client=None,  ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int=3):
        '''Saves audio sample
        
            Args:
                final_waveform(np.array): waveform of processed audio
                ...

            Returns: None
        '''
        if client:
                sf.write('sample_sample.wav', waveform, samplerate=16000, format='WAV')
                client.upload("sample_sample.wav", sample_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
        else: 
                sf.write(sample_path, waveform, samplerate=16000, format='WAV')
         
def create_samples(transcribed_audio:str, main_folder:str, main_audio_wav_path:str, main_audio:str, mode:str, vad_pipeline:Callable, client: Callable=None, ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int= 3 ):
        '''Creating samples of audio...

            Args:
                transcribed_audio(str):
                client (Callable): 

                
            Returns:
                None
        '''
        transcribed_audio_lines= transcribed_audio.split('\n')
        transcribed_audio_lines=[line for line in transcribed_audio_lines ] #if len(line)>3 removed

        #use logger instead of this
        print('всего строк: ',len(transcribed_audio_lines))
        print(transcribed_audio_lines)

        count=0

        for line in transcribed_audio_lines:

            last_number=0
            flag=False

            time_start = str_to_seconds(time_start)
            time_end= str_to_seconds(time_end)
            
            time_start, time_end, name, text = parse_text(line) #maybe will use text to create dataset

            folders_manager = FolderTree(main_folder, 'yandex', participants, ATTEMPTS, ATTEMPTS_INTERVAL, client)

            participant_folder_paths , participants = folders_manager.process_subfolders()

            if not name in participants:
                participants.append(name)
                participant_folder_paths.append(os.path.join(main_folder, name))


            #I should get last_number from here !
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants, mode, client, ATTEMPTS, ATTEMPTS_INTERVAL)

            #taking only relevant sample_audios
            sample_audios = [item for item in sample_audios if item[1]==name]

            last_number=1
            for path, _, folder in sample_audios:
                   number = re.search(r'\d+.wav', path)
                   number = int(number.split('.')[0])
                   if number > last_number:
                           last_number = number
                           
            #Or hm, calcualte 
            if name and flag:

                    #I use numbers because just counting number of files will lead to errors like double files when we delete some files
                    i=last_number 

                    #path hm, default value is name of main audio? if not def., we specify 
                    #what if main_audio downloaded ? then deafault 'downloaded'
                    source_of_sample = main_audio.split('/')[-1].split('.')[0]
                    sample_name = f'{source_of_sample}-{name}_{i+1}.wav'
                    
                    sample_path=os.path.join( folder,  sample_name )

                    #creating audio sample
                    sample_audio = Audio()
                    sample_clip = Segment( time_start , time_end)
                    #'audio.wav'
                    sample_waveform, _ = sample_audio.crop(main_audio_wav_path, sample_clip)
                    sample_waveform=sample_waveform.squeeze(0)
                    sample_waveform=sample_waveform.squeeze(0)

                    # sample_waveform to audio
                    sf.write('cropped_fragment.wav', sample_waveform, samplerate=16000, format='WAV')

                    #removing silent zones from audio sample and saving sample
                    result_of_vad= vad_pipeline('cropped_fragment.wav')
                    time_segments=result_of_vad.get_timeline() # time segments of active speech
                    if len(time_segments)>0:
                                    audio_tool = Audio()
                                    waveforms=[]
                                  
                                    for time_segment in time_segments:
                                        main_clip = Segment( time_segment.start , time_segment.end)
                                        main_waveform, _ = audio_tool.crop('cropped_fragment.wav', main_clip)
                                        main_waveform=main_waveform.squeeze(0)
                                        main_waveform=main_waveform.squeeze(0)
                                        waveforms.append(main_waveform)
                                        
                                    if waveforms:
                                                final_waveform = np.concatenate(waveforms, axis=-1)

                                    # final_waveform to audio
                                    save_sample(sample_path, final_waveform, client)
                    else:
                        save_sample(sample_path, sample_waveform, client)
               
                    #clean trash here! and make more efficient 
                    count+=1
                    print('Saving to:', sample_path)
                    
        os.remove('cropped_fragment.wav')
        print('Fragments added: ', count)

