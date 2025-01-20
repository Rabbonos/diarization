import gdown
import re
import os
from urllib.parse import urlencode #yandex disk
import requests
import shutil
from video import get_timestamps_from_audio, parse_time_range, extract_segment, get_duration, standardize_audio
from typing import List, Optional, Tuple, Callable
from constants import *
import torch
from pyannote.audio import Audio, Pipeline #, Model, Inference
import soundfile as sf
import whisper
import subprocess
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote_whisper.utils import diarize_text ##
import numpy as np
import contextlib
import wave
import json
from typing import Dict
import torchaudio
from yandex import load_audio_from_yandex
from setttings import Settings

def read_json(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
            

def extract_file_id(drive_url):
        """
        Функция для извлечения идентификатора файла из ссылки Google Drive
        """
        # Регулярное выражение для извлечения идентификатора
        file_id = re.search(r'd/([-\w]+)', drive_url)
        if file_id:
            return file_id.group(1)
        else:
            raise ValueError("Не удалось извлечь идентификатор файла из ссылки.")
        

def download_from_gdrive(drive_url, output_path):
                """
                Функция для скачивания файла из Google Drive по ссылке
                """
                try:
                    # Извлекаем идентификатор файла
                    file_id = extract_file_id(drive_url)

                    # Формируем ссылку для скачивания
                    download_url = f"https://drive.google.com/uc?id={file_id}"

                    # Скачиваем файл
                    gdown.download(download_url, output_path, quiet=False)
                    print(f"Файл успешно скачан: {output_path}")

                except Exception as e:
                    raise RuntimeError(f"Произошла ошибка: {e}")
                
##########################################
#settings {source} : None / yandex / google / other url ---> 
#link = file_disk_url
def download_audio(source: str, link: str, download_path: str = 'main_audio.wav') -> str:
        '''
        Downloads audio
        source: str, tells where we got link from , values: [None / yandex / google / other url]
        download_path :str, path where audio is loaded
        '''
        if source == None:
                return
        
        elif source == 'yandex':
                base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
                # get download url
                final_url = base_url + urlencode(dict(public_key=link))
                response = requests.get(final_url)
                download_url = response.json()['href']
                # download & save
            
                download_response = requests.get(download_url)
                with open(download_path, 'wb') as f: 
                    f.write(download_response.content)

                return os.path.abspath(download_path)
        
        elif source =='google':
                download_from_gdrive(link , download_path)
                return os.path.abspath(download_path)
        else:
            raise ValueError('неправильная ссылка на файл! проверьте "file_disk_url"')


def recreate_folder(path: str) -> None:
    """Deletes and recreates a folder."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


#разделение полученных сегментов на более мелкие части (подсегменты)
def fragment(interval: str, main_audio: str, divide_interval: int, temp_folder_main: str, temp_folder_fragments: str) -> List[Tuple] :
        
        """
        Splits audio into smaller segments and subsegments.

        Args:
            intervals (List[str]): List of time intervals (e.g., ["00:00:30-00:01:30"]).
            main_audio (str): Path to the main audio file.
            divide_interval (int): Length of subsegments in seconds.
            temp_folder_main (str): Path to the folder for initial segments.
            temp_folder_fragments (str): Path to the folder for fragmented subsegments.

        Returns:
            List[Tuple[str, int, int]]: Metadata for each subsegment in the format (path, start, end).
        """
                
        #создание первичных отрезков заданных пользователем

        #making folder to store audio
        recreate_folder(temp_folder_main)
        recreate_folder(temp_folder_fragments)

        files_paths = []
        
        #parsing intervals
        # timestamps_text=[]
        # for interval in intervals:
       
        timestamps_text = (parse_time_range(interval, main_audio))

        #extracting audio
        #for start, end in timestamps_text:
            
        start = timestamps_text[0]
        end= timestamps_text[1]
        
        temp_file_name = temp_folder_main+f"/segment{start}_{end}.mp4"

        print(temp_file_name)
        #check out why audio is 60.04 not 60 in length
        # BUG:EXTRACT SEGMENT AND GET_DURATION SHOW DIFFERENCE, EXTRACT_SEGMENT:60S, GET_DURATION:60.04 S!
        extract_segment(main_audio, start, end, temp_file_name)

        temp_audio_path = [temp_file_name, start, end]
        files_paths.append(temp_audio_path)

        #обработка сегментов

        timestamps_audio_list= []

        #создание временных меток в формате [путь, [временные метки], начало основного сегмента, конец основного сегмента]
        for i, f_path in enumerate(files_paths):

            temp_audio_path = [f_path[0], get_timestamps_from_audio(f_path[0], divide_interval), f_path[1], f_path[2]]
            timestamps_audio_list.append(temp_audio_path)

        print("Подсегменты")
        subsegments  = []
        audio_len_sec = get_duration (main_audio)

        for el in timestamps_audio_list:
            segment_path = el[0]
            for timestamp in el[1]:

                #время начала и конца подсегмента
                start=timestamp[0]
                end = timestamp[1]
                end = min(end, int(audio_len_sec) )

                print(start, end, 'START AND END')

                audio_mav_name = segment_path.split('/')[-1].split('mav')[0][7:-1:]+"___"
                temp_file_name = temp_folder_fragments+f"/subsegment{audio_mav_name}{start}_{end}.mp4"    #уникальное имя файла .mav для подсегмента

                #создание временных меток в формате [путь,  начало подсегмента со сдвигом , конец подсегмента со сдвигом]
                temp_video_path = [temp_file_name, start, end]

                # сегмент для извлечения (с какой по какую минуту)
                extract_segment(segment_path, start, end, output_path=temp_file_name)
                subsegments.append(temp_video_path)

        return subsegments #[fragment path, start time, end time]


#длина голосов
def preprocess_segments(segments, min_duration=1):
        return [segment for segment in segments if (segment['end'] - segment['start']) >= min_duration]


def delete_overlapped_segments(old_segments, new_segments):
                    # Create a list to store non-overlapped segments
                    filtered_segments = []

                    # Sort both old and new segments by start time
                    old_segments = sorted(old_segments, key=lambda x: x['start'])
                    new_segments = sorted(new_segments, key=lambda x: x['start'])

                    # Iterate through old segments
                    for old_seg in old_segments:
                        # Assume the segment is kept until proven overlapped
                        is_overlapped = False

                        # Check against each new segment
                        for new_seg in new_segments:
                            # Check if old segment is completely contained within a new segment
                            if (new_seg['start'] <= old_seg['start'] and new_seg['end'] >= old_seg['end']):
                                is_overlapped = True
                                break

                        # Only keep the segment if it's not overlapped
                        if not is_overlapped:
                            filtered_segments.append(old_seg)

                    return filtered_segments


# Загрузка модели Whisper должнa быть до model = load_model...
def load_model(language: str, model_size: str):
    model_name = model_size
    if language == 'English' and model_size != 'large' and model_size != 'turbo': #Укажите модель Whisper (tiny, base, small, medium, large, large-v2)
        model_name += '.en'

    current_model = whisper.load_model(model_name)
    print(f'Loaded {model_name} model')
    return current_model



def load_desilencer(settings, device:str)->Callable:
            """
            Loads the VAD pipeline to remove silent zones.
            """
            vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                              use_auth_token=settings.HUGGINGFACE_TOKEN ).to(device) #модель для удаления тишины
            return vad_pipeline


class AudioProcessor:
        def __init__(self, settings: Settings  ):
            """
            Initializes the audio processing system.
            """
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # девайс если есть, то gpu
            print(f"Using device: {self.device}")
            self.settings = settings

            # load whisper model
            self.model = load_model(self.settings.language, self.settings.modeltype).to(self.device) 

            if self.settings.Silence:
                # Load VAD pipeline
                self.vad_pipeline = load_desilencer(self.settings,self.device)

            if self.settings.Accuracy_boost:
                self.diarization_pipeline = self._load_accuracy_boost()

            self.embedding_model, self.embedding_model_dimension, self.speaker_model = self._setup_embedding_model()

    
        def _load_accuracy_boost(self):
            """
            Loads the speaker diarization pipeline to improve accuracy.
            """
            if self.settings.Accuracy_boost:
                  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                use_auth_token=self.settings.HUGGINGFACE_TOKEN).to(self.device)
            return pipeline 
        

        def _setup_embedding_model(self):
            """
            Loads the embedding model based on the selected type.
            """
            speaker_model=None
            #global embedding_model , speaker_model
            if self.settings.embedding_model_name in self.settings.embedding_models_group1:
                    #модель эмбеддингов группы 1 (выбор: 'pyannote/embedding', 'speechbrain/spkrec-ecapa-voxceleb', 'nvidia/speakerverification_en_titanet_large' или 'hbredin/wespeaker-voxceleb-resnet34-LM' ) вставить ваш выбор + embedding_model_dimension= 512 для 'pyannote/embedding' и embedding_model_dimension= 192 для 'speechbrain/spkrec-ecapa-voxceleb' , для hbredin/wespeaker-voxceleb-resnet34-LM embedding_model_dimension= 512 , для 'nvidia/speakerverification_en_titanet_large' embedding_model_dimension= 192
                    embedding_model = PretrainedSpeakerEmbedding( self.settings.embedding_model_name, device=self.device,
                                                                  use_auth_token=self.settings.HUGGINGFACE_TOKEN)

            elif self.settings.embedding_model_name in self.settings.embedding_models_group2:
                    try:
                        import nemo.collections.asr as nemo_asr
                    except ImportError as e:
                        raise ImportError(
                            "Nemo is required for this feature. Install it using: pip install ...[nemo]."
                        ) from e
                    #модель эмбеддингов группы 2 (выбор : 'titanet_large' , 'ecapa_tdnn' , 'speakerverification_speakernet') , embedding_model_dimension = X (искать в интернете/chatgpt)  , для titanet_large embedding_model_dimension=192 , для speakerverification_speakernet - 256 ...
                    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.settings.embedding_model_name)
                    speaker_model = speaker_model.to(self.device)

                    def embedding_model(speaker_model, audio_waveform):
                        # Create a temporary file manually
                        audio_waveform=audio_waveform.squeeze(0)
                        audio_waveform=audio_waveform.squeeze(0)
                        temp_file_path = "temp_audio_file.wav"

                        # Write the audio waveform to the file
                        sf.write(temp_file_path, audio_waveform, samplerate=16000)

                        try:
                            # Process the file with the speaker model
                            result = speaker_model.get_embedding(temp_file_path)
                        finally:
                            # Ensure the file is deleted after processing
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)

                        return result
            else:
                    raise ValueError('Incorrect model name')
    
            embedding_model_dimension = self.settings.EMBEDDING_MODELS[self.settings.embedding_model_name]

            return embedding_model, embedding_model_dimension , speaker_model
        

        def remove_overlap(self,main_audio):
            """
            Processes overlapping speech segments and extracts timestamps and audio clips.
            """
            #находим отрезки с пересечением голосов продолжение в def process_file...
            
            # Load the overlapped speech detection pipeline
            osd_pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection", 
                                                    use_auth_token=self.settings.HUGGINGFACE_TOKEN)

            #NOT REALLY CORRECT? I SHOULD RUN IT ON FRAGMETNS
            # ALSOO IDEALLY FRAGEMTNS SHOULD BE SPLIT IN A WAY WHERE SPEECH ENDS, NOT BY SETTING? OR SETTING+ENDING...
            main_wav='main_wav.wav'
            subprocess.call(['ffmpeg', '-i', main_audio, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-y', main_wav])#'-frame_size', '400', 
            
            overlap_segments = osd_pipeline({"audio": main_wav})

            # Ensure the output folder exists
            os.makedirs(self.settings.CLIPPED_SEGMENTS, exist_ok=True)

            audio = Audio()

            overlap_timestamps=[]
            # Print overlap segments and clip them
            for i, segment in enumerate(overlap_segments.itertracks(yield_label=False)):
                    segment=segment[0]
                    print(f"Overlap detected from {segment.start:.2f}s to {segment.end:.2f}s")
                    overlap_timestamps.append([segment.start,segment.end ])
                    # Clip the segment
                    clipped_audio = Segment(segment.start, segment.end)
                    audio_waveform, _ = audio.crop(main_wav, clipped_audio)

                    # Save the clipped audio to a file
                    output_path = os.path.join(self.settings.CLIPPED_SEGMENTS, f"overlapped_segment_{i+1}.wav")

                    audio_waveform= audio_waveform.squeeze(0)
                    sf.write(output_path, audio_waveform.numpy(), 16000)
                    print(f"Saved segment {i+1} from {segment.start}s to {segment.end}s as {output_path}")

            return overlap_timestamps


def desilence(wav_path:str, vad_pipeline:Callable)->None:
        """
        Removes silent regions from an audio file to improve transcription accuracy.

        Args:
            wav_path (str): Path to the WAV file.
            vad_pipeline (Callable): Voice Activity Detection pipeline.
        """
        try:
            result_of_vad= vad_pipeline(wav_path)
            time_segments=result_of_vad.get_timeline() # time segments of active speech

            main_wave_audio = Audio()
            waveforms=[]

            with contextlib.closing(wave.open(wav_path, 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)

            for time_segment in time_segments:
                    speech_start = time_segment.start
                    speech_end = min(duration, time_segment.end)
                    main_clip = Segment( speech_start , speech_end)
                    main_waveform, _ = main_wave_audio.crop(wav_path, main_clip)
                    main_waveform=main_waveform.squeeze(0)
                    main_waveform=main_waveform.squeeze(0)
                    waveforms.append(main_waveform)

            if waveforms:
                    final_waveform = np.concatenate(waveforms, axis=-1)
                    sf.write(wav_path, final_waveform, samplerate=16000, format='WAV')
        except Exception as e:
              raise RuntimeError(f"Error during desilencing: {e}")

def boost_accuracy(pipeline:Callable, wav_path:str, result:dict)->List[dict]:
        """
        Boosts transcription accuracy using speaker diarization.

        Args:
            pipeline (Callable): Diarization pipeline.
            wav_path (str): Path to the WAV file.
            result (Dict): Transcription result.

        Returns:
            List[Dict]: Enhanced transcription segments
        """
        try:
            segments=[]
            diarization_result = pipeline(wav_path)
            final_result = diarize_text(result, diarization_result)

            for segment, speaker, sentence in final_result:
                segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": sentence,
                "speaker": speaker
                                })
                
            return segments
        except Exception as e:
            raise RuntimeError(f"Error during accuracy boosting: {e}")
            


#redundancy with is_fragment, because I will anyway need fragments then
def main_audio_preprocessing(model:Callable, wav_path:str, is_fragment:str ,main_audio:str=None, fragments:List[List]=None, Silence:bool=False , Accuracy_boost:bool=False, vad_pipeline:Optional[Callable]=None, pipeline: Optional[Callable]=None)->dict:
                    """
                    Preprocesses audio data for transcription, can include silence removal and accuracy boosting.

                    Args:
                        fragments (List[List]): List of [path, start, end].
                        model (Callable): Transcription model.
                        is_fragment (str): Whether the input consists of fragments.
                        silence (bool): Flag for enabling silence removal.
                        accuracy_boost (bool): Flag for enabling accuracy boosting with diarization.
                        vad_pipeline (Optional[Callable]): Voice Activity Detection pipeline.
                        pipeline (Optional[Callable]): Speaker diarization pipeline.

                    Returns:
                        Dict: Processed transcription result.
                    """
                    
                    if is_fragment!='да':
                          standardize_audio( main_audio, wav_path)

                          if Silence:
                                desilence(wav_path, vad_pipeline)

                          result = model.transcribe(wav_path)
                          result["segments"] = preprocess_segments(result["segments"])

                          if Accuracy_boost:
                                  result["segments"] = boost_accuracy(pipeline, wav_path, result)

                          return result
                    else:
                          result={}
                          for path, start, end in fragments:
                              print(f"обработка {path}")
                              subprocess.call(['ffmpeg', '-i', path, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-y', wav_path]) #'-frame_size', '400',

                              if Silence:
                                 desilence(wav_path, vad_pipeline)

                              temp_result = model.transcribe(wav_path)

                              if not result:
                                  result=temp_result
                              else:
                                  id = result['segments'][-1]['id']+1
                                  for i in enumerate(temp_result['segments']):
                                        temp_result['segments'][i]['id'] = id
                                        result['segments'].append(temp_result['segments'][i])
                                        id+=1

                              result["segments"] = preprocess_segments(result["segments"])
                              if Accuracy_boost:
                                  result["segments"] = boost_accuracy(pipeline, wav_path, result)

                              return result
                          
def get_sample_embeddings(
        settings: Settings,
        sample_audios: List, 
        sample_vectors: List, 
        embedding_model: Callable, 
        speaker_model: Callable, 
        client:Callable=None ):
        """
        Processes audio samples, extracts embeddings, and updates vector storage.

        Args:
            settings(Settings): Dataclass instace that stores all settings
            sample_audios (List): List of audio samples.
            sample_vectors (List): List of vectors to be updated.
            embedding_model (Callable): Model for extracting embeddings.
            speaker_model (Callable): Model for speaker embedding.
            embedding_models_group1 (List[str]): Group of embedding models.
            sample_embeddings (List): List to store embeddings.
            client (Callable): Client for downloading/uploading.
            ATTEMTPS (int): Number of retries for downloading/uploading.
            ATTEMPTS_INTERVAL (int): Interval between retries in seconds.
        """
        sample_embeddings=[]
        for sample_path, name, folder_name in sample_audios:
            if client:
                        sample_path_out= os.path.join(os.path.getcwd(), 'sample_audio.wav')
                        load_audio_from_yandex(sample_path, client, sample_path_out)
                        sample_path = sample_path_out

            dict_for_json={}
            sample_audio = Audio()
            sample_wav_path = f'sample_voice.wav'

            #standardize audio
            standardize_audio(sample_path, sample_wav_path )
            sample_duration = get_duration(sample_wav_path)

            # Process audio if duration > 1 second
            if sample_duration>1:
                    sample_clip = Segment(0, sample_duration)
                    sample_waveform, _ = sample_audio.crop(sample_wav_path, sample_clip)

                    if settings.embedding_model_name in settings.embedding_models_group1:
                        sample_embedding= embedding_model(sample_waveform[None])
                    else:
                        sample_embedding=embedding_model(speaker_model, sample_waveform[None]).cpu()

                    sample_embeddings.append([ sample_embedding, name ] )
                    dict_for_json[f'{sample_path}']=[sample_embedding.tolist()]

                    if settings.Add:
                        add_vectors(settings, sample_vectors, client, dict_for_json, 'downloaded.json')

            return sample_embeddings
        

def replace_overlaps(segments:List[Dict], sep_model:Callable, model:Callable, overlap_timestamps:List[List[float]], 
                     clipped_segments_dir: str = "./clipped_segments",
                     output_dir: str = "./voices_split" )->List[Dict]:
            """
            Replaces overlapping audio segments with new transcribed segments of separated voices.

            Args:
                sep_model (Callable): Model to separate voices from audio.
                model (Callable): Whisper model to transcribe audio.
                overlap_timestamps (List[List[float]]): List of [start, end] timestamps for overlaps.
                clipped_segments_dir (str): Directory containing clipped audio segments.
                output_dir (str): Directory to save separated voice waveforms.

            Returns:
                List[Dict]: Processed and updated segments of transcribed audio.
            """
            os.makedirs(output_dir, exist_ok = True)
            #delete old segments and replace them with new ones below
            for overlapped_segment, timestamp in zip(os.listdir(clipped_segments_dir), overlap_timestamps):
                        overlapped_segment_path= os.path.join(clipped_segments_dir, overlapped_segment)
                        output_waveforms = sep_model.separate_file(overlapped_segment_path)
                        length = output_waveforms[:, :, :].shape[2]
                        # Temporary list to collect new segments
                        new_segments = []                      
                        # Process the separated waveforms
                        for i in range(length):
                            torchaudio.save(f"./{output_dir}/separated_voice.wav", output_waveforms[:, :, i].detach().cpu(), 16000)
                            transcribed_segments = model.transcribe(f"./{output_dir}/separated_voice.wav")["segments"]
                            #Add timestamp (otherwise it will start from 00:00:00 each time)
                            for segment in transcribed_segments:
                                segment['start'] = timestamp[0]
                                segment['end'] = timestamp[1]
                                new_segments.append(segment)
                        # Remove overlapped old segments
                        segments = delete_overlapped_segments(segments, new_segments)
                        segments.extend(new_segments)
            return segments #unordered


          

def use_vectors(settings: Settings,sample_vectors:List[List], client=None, output_path:str = "downloaded.json" )-> List[List]:
            '''Using vectorised samples instead of sample audios
            
                Args:
                    sample_vectors(List[List[str, str, str]]): embeddigns of sample audios
                    mode(str): mode of operation ['local','yandex','google']
                    ATTEMPTS(int) : how many tries at sending yandex a request
                    ATTEMPTS_INTERVAL(int) : intervals for attemots
                    output_path (str): path of output json (required only if 'yandex' is used)
                Returns:
                    sample_embeddings(List[List[np.ndarray, str]]): embeddigns of sample audios
            '''
            sample_embeddings = []
            for json_path, name, folder in sample_vectors:
                    if client:
                            try:
                                client.download(
                                    json_path, output_path, 
                                    n_retries=settings.ATTEMPTS, retry_interval=settings.ATTEMPTS_INTERVAL
                                )
                                json_path = output_path
                            except Exception as e:
                                raise RuntimeError(f"Failed to download JSON file from '{json_path}': {e}")
                    try:
                        embedding_sample=read_json(json_path) # 'path' : '132481832471823812', ... 
                    except:
                        print(f'It seems that json file "{json_path}" is Empty')
                        continue
                    for key, value in embedding_sample.items():
                        try:
                            #Convert string to NumPy array
                            embedding = np.array(value[0], dtype=float)
                            sample_embeddings.append([embedding, name])
                        except (ValueError, IndexError) as e:
                             print(f"Error processing embedding for key '{key}' in file '{json_path}': {e}")
                             
            return sample_embeddings



def add_vectors(settings: Settings, sample_vectors:List, client:Callable, dict_for_json:dict, output_path:str="downloaded.json"):
            """
            Converts audio to vectors and stores them in a JSON file.

            Args:
                sample_vectors (List): List of sample vectors to be updated.
                client (Callable): Client for uploading and downloading JSON files.
                dict_for_json (dict): Data to be added to the JSON file.
                output_path (str): Temporary local path for the JSON file.
                ATTEMTPS (int): Number of retries for downloading/uploading.
                ATTEMPTS_INTERVAL (int): Interval between retries in seconds.
            """
            for json_path, name , folder_name in sample_vectors:
                    if client:
                            client.download(json_path, output_path, n_retries = settings.ATTEMPTS  , retry_interval =settings.ATTEMPTS_INTERVAL)
                            json_path_copy= json_path
                            json_path= os.path.join(os.path.getcwd(), output_path)
                            
                    if os.path.exists(json_path):
                        vectorized_sample_old =read_json(output_path)
                        vectorized_sample_old.update(dict_for_json)
                        with open(json_path, 'w') as f:
                                json.dump(vectorized_sample_old, f, indent=4)
                        if client:
                                client.upload(json_path, json_path_copy, n_retries = settings.ATTEMPTS  , retry_interval =settings.ATTEMPTS_INTERVAL)
                    else:
                          raise NameError(f'{json_path} does not exist, probably downloading that from yandex failed' )
                    




def segment_embedding(
                        settings: Settings,
                        segment: dict, 
                        embedding_model: Callable, 
                        wav_path: str, 
                        embedding_model_dimension: int, 
                        duration: float, 
                        speaker_model: Callable ):
                        """
                        Extracts embeddings for a single audio segment.

                        Args:
                            segment (dict): Dictionary with segment information containing "start" and "end".
                            embedding_model (Callable): Model for extracting embeddings.
                            wav_path (str): Path to the WAV file.
                            embedding_model_dimension (int): Dimension of the embedding vector.
                            duration (float): Duration of the WAV file.
                            speaker_model (Callable): Model for speaker embedding.
                        

                        Returns:
                            np.ndarray: Embedding vector for the segment.
                        """
                        audio = Audio()
                        start = segment["start"]
                        end = min(duration, segment["end"])
                        
                        # Check if the segment has a valid duration
                        if end <= start:
                            print(f"Warning: Skipping zero-length segment: {start} to {end}")
                            return np.zeros(embedding_model_dimension)
                        
                        clip = Segment(start, end)
                        waveform, sample_rate = audio.crop(wav_path, clip)
                        # Check if the waveform is empty
                        if waveform.shape[1] == 0:
                            print(f"Warning: Empty waveform for segment: {start} to {end}")
                            return np.zeros(embedding_model_dimension)

                        if settings.embedding_model_name in settings.embedding_models_group1:
                            return embedding_model(waveform[None])
                        else:
                            return embedding_model(speaker_model, waveform[None]).cpu()

def get_embeddings_main_audio(
            settings: Settings,
            wav_path: str, 
            segments: List[dict], 
            embedding_model: Callable, 
            embedding_model_dimension: int, 
            speaker_model: Callable, 
                    ):
                    """
                    Extracts embeddings for all segments in a main audio file.

                    Args:
                        wav_path (str): Path to the WAV file.
                        segments (List[dict]): List of segment dictionaries containing "start" and "end".
                        embedding_model (Callable): Model for extracting embeddings.
                        embedding_model_dimension (int): Dimension of the embedding vector.
                        speaker_model (Callable): Model for speaker embedding.

                    Returns:
                        np.ndarray: Array of embeddings for all segments.
                    """
                    duration = get_duration(wav_path)
                    embeddings = np.zeros(shape=(len(segments), embedding_model_dimension))

                    for i, segment in enumerate(segments):
                      if settings.embedding_model_name in settings.embedding_models_group1:
                        embeddings[i] = segment_embedding(
                                                          settings,
                                                          segment, 
                                                          embedding_model, 
                                                          wav_path,embedding_model_dimension,
                                                          duration, 
                                                          speaker_model  )
                      else:
                        embeddings[i] = segment_embedding(
                                                          settings,
                                                          segment, 
                                                          embedding_model,
                                                          wav_path,
                                                           embedding_model_dimension,
                                                            duration, speaker_model ).cpu()
                    
                    return embeddings