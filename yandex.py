import yadisk
import json
import os
from typing import Callable, Dict

def get_new_ytoken(client_id:str, client_secret:str):
        
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


def handle_yandex_json(client:Callable, json_path:str, name:str, dict_for_json: Dict,  ATTEMPTS:int=3, ATTEMPTS_INTERVAL:int=3):
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
    


def load_audio_from_yandex(sample_path:str , client:Callable, output_path:str = 'sample_audio.wav' , ATTEMTPS:int=3, ATTEMPTS_INTERVAL:int=3):
        """
        Downloads audio from Yandex and saves it locally.

        Args:
            sample_path (str): Path to the audio file on Yandex.
            client (Callable): Client for downloading the audio.
            output_path (str): Local path to save the downloaded audio.
            ATTEMTPS (int): Number of retries for downloading.
            ATTEMPTS_INTERVAL (int): Interval between retries in seconds.

        Raises:
            RuntimeError: If downloading fails.
        """
        if client:
            try:
                client.download(sample_path, output_path, n_retries = ATTEMTPS  , retry_interval =ATTEMPTS_INTERVAL)
            except Exception as e:
                raise RuntimeError(f"During downloading audio from yandex an exception happened {str(e)}")