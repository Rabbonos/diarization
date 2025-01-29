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
#TO ADD: async stuff + CACHING!
# from google.colab import files
# from google.colab import userdata


#IMPORTANT , setting silence  + sampling do not work , silence removes some audio and time is reduced, sampling does not take that into account OR something like that

from transcribe import start
from vectorize import upgrade_vectors
from sampling import create_samples
import yaml
from settings import ClusterSettings, Settings


#handle potential errors
class Diarizator:
      '''
      Main object of library, whatever the library is 
      capable of (not a lot) is handled here
      '''
      def __init__(self, path):
            self.settings=self.import_settings(path)
            self.vad_pipeline=None
            self.client=None
            self.text=None
      def import_settings(self, path:str):
            with open(path, 'r', encoding='utf-8') as f:
                   settings = yaml.safe_load(f)
            cluster_settings_data = settings.pop('cluster_settings')
            settings = Settings(**settings, cluster_settings=ClusterSettings(**cluster_settings_data))
            
            return settings

      def transcribe(self):
            vad_pipeline, client= start(self.settings)
            self.vad_pipeline= vad_pipeline
            self.client= client

      def sample(self):
            #main audio  wav path
            create_samples(self.settings, self.text, self.client ,self.vad_pipeline)
      
      def upgrade_vectors(self):
            upgrade_vectors(self.settings)
      
      def display_settings(self):
            print(self.settings)

if __name__ == '__main__':

      diarizator = Diarizator('settings.yaml')
      diarizator.text = '''
[0:00:58,240 --> 0:01:00,000] Mr.X X:  Перчаточки это у меня.'''
      diarizator.transcribe()
      #diarizator.sample()
      #diarizator.upgrade_vectors()

#in sampling i might have group 0, neewd to take into accoutna an empty line!