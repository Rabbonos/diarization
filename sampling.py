import re
from file_manager import FolderTree
import os
from settings import Settings
from file_manager import fetch_vectors_and_audio_files
import soundfile as sf
from pyannote.audio import Audio
from pyannote.core import Segment
import numpy as np
import torch
from audio_processing import load_desilencer
import yadisk
import librosa
from video import get_duration
#00:00:00-00:01:00

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
        pattern_name = re.compile('(?<=\])\s*(.*?)(?=:)\s*')
        pattern_text = re.compile('(?<=:) .*')

        #names don't have digits!!!??!?
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
         
def create_samples(settings: Settings, transcribed_audio:str, client=None,  vad_pipeline=None ):
        '''Creating samples of audio...

            Args:
                transcribed_audio(str):
                client (Callable): 
                
            Returns:
                None
        '''
        if settings.mode=='yandex':
               client = yadisk.Client(token=settings.token_yandex  )
        #already find device in audio_processing, correct it later
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not vad_pipeline:
                vad_pipeline = load_desilencer(settings, device )
        
        transcribed_audio_lines= transcribed_audio.split('\n')
        transcribed_audio_lines=[line for line in transcribed_audio_lines ] #if len(line)>3 removed

        #use logger instead of this
        print('всего строк: ',len(transcribed_audio_lines))
        print(transcribed_audio_lines)

        count=0

        for line in transcribed_audio_lines:

            if line == '':
                    continue
            
            last_number=0

            #temp solution
            try:
                time_start, time_end, name, text = parse_text(line) #maybe will use text to create dataset
            except:
                   continue
            
            time_start = str_to_seconds(time_start)
            time_end= str_to_seconds(time_end)
            

            folders_manager = FolderTree(settings.main_folder, settings.PARTICIPANTS, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL, client)

            participant_folder_paths , participants = folders_manager.process_subfolders()
            participant_folder_paths , participants = list(participant_folder_paths) , list(participants)
            if not name in participants:
                participants.append(name)

            folders_manager = FolderTree(settings.main_folder, participants, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL, client)
            participant_folder_paths , participants = folders_manager.process_subfolders()
            participant_folder_paths , participants = list(participant_folder_paths) , list(participants)

            #I should get last_number from here !
            sample_vectors, sample_audios =  fetch_vectors_and_audio_files(participant_folder_paths, participants,client, settings.ATTEMPTS, settings.ATTEMPTS_INTERVAL)

            #taking only relevant sample_audios
            sample_audios = [item for item in sample_audios if item[1]==name]
            sample_vectors=  [item for item in sample_vectors if item[1]==name]
            folder= sample_vectors[-1][-1]
            last_number=0
            for path, _, folder in sample_audios:
                   number = re.search(r'\d+.wav', path)
                   number=number.group(0)
                   number = int(number.split('.')[0])
                   if number > last_number:
                           last_number = number
            
            #Or hm, calcualte 
            if name:

                    #I use numbers because just counting number of files will lead to errors like double files when we delete some files
                    #path hm, default value is name of main audio? if not def., we specify 
                    #what if main_audio downloaded ? then deafault 'downloaded'
                    source_of_sample = settings.main_audio.split('/')[-1].split('.')[0]
                    sample_name = f'{source_of_sample}-{name}_{last_number+1}.wav'
                    
                
                    sample_path=os.path.join( folder,  sample_name )
                    #temporary solution
                    if client:
                           sample_path=sample_path.replace('\\','/')

                    
                    #creating audio sample
                    sample_audio = Audio()
                    sample_clip = Segment( time_start , time_end)

                    data, sr = librosa.load(settings.main_audio_wav_path, sr=16000)
                    #temp solution!
                    try:  # Stereo case
                        data = data.T  # Transpose to (frames, channels)
                        sf.write(settings.main_audio_wav_path, data, sr)
                    except: pass
                    sample_waveform, _ = sample_audio.crop(settings.main_audio_wav_path, sample_clip)
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
                                    duration = get_duration('cropped_fragment.wav')
                                    for time_segment in time_segments:
                                        #temp solution!
                                        end=min(time_segment.end, duration)
                                        main_clip = Segment( time_segment.start, end)
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

        #temp solution
        try:   
              os.remove('cropped_fragment.wav')
        except:
              pass
        print('Fragments added: ', count)

