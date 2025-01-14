import os
from constants import ATTEMPTS, ATTEMPTS_INTERVAL
import unicodedata
import warnings
import json
from typing import Tuple, List, Callable

#might lose superscripts and other certain equivalents
def normalize_string(s):
        return unicodedata.normalize('NFKC', s)



#silly, inefficient way know, checks if 2 words are the same in folder, then it is our folder
class FolderTree():
        '''
        
        Retrieves a list of subfolders based on the client.

        Args:
            client (str): client of operation ('yandex' or others).
            client: Client object for remote operations.
            attempts (int): Number of retries for remote operations.
            attempts_interval (int): Interval between retries in seconds.

        Returns:
            list: List of subfolder paths.

        '''
        def __init__(self, folder_path:str, participants:List[str], ATTEMPTS=3, ATTEMPTS_INTERVAL= 3,  client=None, ):
            """initializes folders and subfolders"""
            self.client = client
            self.folder_path = folder_path
            self.participants = participants
            self.ATTEMPTS = ATTEMPTS
            self.ATTEMPTS_INTERVAL = ATTEMPTS_INTERVAL 
            self.subfolders = self._get_subfolders ()

        def _get_subfolders(self):
               """gets list of subfolders based on the client"""
               if self.client:
                    try:
                        self.subfolders = self.client.listdir(self.folder_path, n_retries = self.ATTEMPTS  , retry_interval = self.ATTEMPTS_INTERVAL)
                        return [x['path'] for x in self.subfolders]
                    except Exception as e:
                           raise RuntimeError(f"Failed to get subfolders:{e}")                
               else:
                    if not os.path.exists(self.folder_path):
                           raise FileNotFoundError(f"Folder was not found, path given:{self.folder_path}")
                    return [os.path.abspath(folder) for folder in os.listdir(self.folder_path)]


        def create_folder(self, folder_path):            
               """Creates a folder with respect to client.

                  Args:
                        folder_path (str): abs. path to a folder
               """
               if self.client:
                    self.client.mkdir(folder_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
               else:
                    os.mkdir(folder_path)

        def process_subfolders(self):

            """Match participants to their folders and create missing folders."""
            
            # Step 2: Create a list of words and corresponding folder names for sample folders
            temp_list_folder = []
            for participant_folder_path in self.subfolders:
                
                person_name_split = normalize_string(os.path.basename(participant_folder_path)).split(' ') # Split folder name into words
                temp_list_folder.append([person_name_split, participant_folder_path])  # Store words and name
            
            # Step 3: Create a list of words and corresponding participant names
            tem_list_participant = []
            for participant_name in self.participants:
                participant_words = normalize_string(participant_name.strip()).split(' ')
                tem_list_participant.append([participant_words, participant_name])  # Store words and name

            # Step 4: Compare the lists and find matches
            matched_pairs =[]
            for participant_name_split, participant_name in tem_list_participant:
                match_found=False
                for participant_name_split_, participant_folder_path in temp_list_folder:
                    if len(set(participant_name_split_).intersection(set(participant_name_split))) >= 2:  # At least 2 matching words
                        matched_pairs.append([participant_folder_path, participant_name ])
                        match_found=True
                        break
                if not match_found:
                    name=name.strip()
                    folder= os.path.join (self.folder_path, ' '.join(participant_name_split))
                    warnings.warn(f'for participant {name} folder was not found, creating a new folder:{ folder }')
                    self.create_folder(folder)
                    matched_pairs.append([folder, participant_name])

            participant_folder_paths, participants = zip(*matched_pairs)

            return participant_folder_paths , participants
        


def fetch_vectors_and_audio_files(participant_folder_paths:List[str], participants:List[str], client:Callable=None, ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int= 3)->Tuple[List[List[str]], List[List[str]]]:
            """
            Collects sample vector and audio file data for participants.

            Args:
                participant_folder_paths (list): List of folder paths for participants.
                participants (list): List of participant names.
                client (object, optional): Client object for interacting with remote files in 'yandex' mode.
                attempts (int, optional): Number of retries for client operations.
                attempts_interval (int, optional): Interval between retries.

            Returns:
                tuple: Two lists - sample_vectors and sample_audios.
                    - sample_vectors: List of [file_path, participant_name, folder_path] for vector files.
                    - sample_audios: List of [file_path, participant_name, folder_path] for audio files.
            """
            def listing_files(folder:str, client:Callable=None, ATTEMPTS=3, ATTEMPTS_INTERVAL= 3)->list:
                   """Lists files in the folder based on the specified mode."""
                   if client:
                          with client:
                                return client.listdir(folder, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL)
                   else:
                          return os.listdir(folder)     
                              
            def create_vector_file(empty_json_path:str,  client:Callable=None, ATTEMPTS=3, ATTEMPTS_INTERVAL= 3):
                    """Creates an empty JSON file for a participant if none exists."""
                    if client:
                         with client:
                                client.upload(empty_json_path, empty_json_path, n_retries = ATTEMPTS  , retry_interval =ATTEMPTS_INTERVAL )
                    else:                        
                            with open(empty_json_path, 'w') as f:
                                            json.dump({}, f)

            def check_file_type(folder, file, meta, client:Callable=None) -> bool:
                   """Checks if a file is valid (not a directory)."""
                   try:
                        if client:
                            return meta['type'] != 'dir'
                        return os.path.isfile(os.path.join(folder, file))
                   except KeyError as e:
                        raise KeyError(f"Unexpected error: {e}")
             
            sample_vectors = []
            for folder,name in zip(participant_folder_paths, participants):
                                files = listing_files(folder, client, ATTEMPTS, ATTEMPTS_INTERVAL )
                                for file in files:
                                            file= file['path'] if client!=None else os.path.join(folder, file) 
                                            if file.endswith('.json'):
                                                    sample_vectors.append([file, name, folder])
                                                    
                                if  not any(name == vector[1] for vector in sample_vectors) :
                                        empty_json_path = os.path.join(folder, f"{name}.json")
                                        create_vector_file(empty_json_path, client, ATTEMPTS, ATTEMPTS_INTERVAL)
                                        sample_vectors.append([empty_json_path, name, folder])

            sample_audios = []
            for folder,name in zip(participant_folder_paths, participants):
                            files = listing_files(folder, client, ATTEMPTS, ATTEMPTS_INTERVAL )
                            for file in files:
                                meta=file
                                file= file['path'] if client!=None else os.path.join(folder, file) 
                                if not file.endswith('.json') and  check_file_type(folder, file, meta, client): 
                                    sample_audios.append([file, name, folder])

            return (sample_vectors, sample_audios)


