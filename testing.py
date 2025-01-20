import re
from typing import List, Union


text = "|00:01:30-00:02:30|"
text2="|00:01:30-end|"

def parse_time_range(text:str, audio_len:int)-> tuple[int,int]:
        """
        Parses a time range from a given string and converts it to seconds.
        
        Args:
            text (str): Input string containing the time range.
            audio_len (int): The total audio length in seconds.

        Returns:
            tuple[int, int]: Start and end times in seconds.
        """

        TIME_PATTERN =r"(?<=\|)\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}(?=\|)|(?<=\|)\d{2}:\d{2}:\d{2}-end(?=\|)"
        CONVERTER_TO_SEC=[3600,60,1]

        match=re.findall(TIME_PATTERN, text)
        if not match:
                raise ValueError(f'No valid time range found in text:{text}')
        time_range=match[0]
        start_str, end_str= time_range.split('-')
        start_parts = [int(x) for x in start_str.split(':')]
        start_time = sum(digit * seconds for digit, seconds in zip(start_parts, CONVERTER_TO_SEC))

        if end_str == 'end':
                 end_time = audio_len
        else:
             try:
                 end_parts = [int(x) for x in end_str.split(':')]
                 end_time = sum(digit * seconds for digit, seconds in zip(end_parts, CONVERTER_TO_SEC))
             except:
                     raise ValueError(f'Wrong time range in : {text}')
             
        return start_time, end_time

#start_time, end_time = parse_time_range(text2, 100 )
#print(start_time, end_time)
# words=['ali','isayev']
# print(' '.join(words))

#pip (poetry add) install python-docx


# write_to_word('hi\n my name is Ali\n And your?', r'C:\Users\Ali\Desktop\newword.docx')
# import torch
# print(torch.cuda.is_available())
# print(torch.version.cuda)


import librosa
import soundfile as sf

data, sr = librosa.load("main_audio.wav", sr=None, mono=False)
print(data.ndim)
if data.ndim == 2:  # Stereo case
    data = data.T  # Transpose to (frames, channels)
sf.write("output_file.wav", data, sr)