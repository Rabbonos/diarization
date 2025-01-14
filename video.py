import subprocess
import re

def extract_segment(main_audio, start, end, output_path):
                """
                Extracts audio segment from the file

                main_audio: str
                start: float (seconds)
                end: float (seconds)
                output_path: str
                """
                duration = end - start

                # Construct the FFmpeg command
                command = [
                    "ffmpeg",
                    "-ss", str(start),  # Start time in seconds
                    "-i", main_audio,  # Input video file
                    "-t", str(duration),  # Duration of the clip in seconds
                    "-c", "copy",  # Copy codec (no re-encoding)
                    output_path  # Output file
                ]

                # Run the command using subprocess
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Check the output and errors
                if result.returncode != 0:
                    raise RuntimeError(f"Error: {result.stderr.decode()}")
                else:
                    print(f"Segment extracted successfully: {output_path}")

def get_duration(file_path):
        """
        Get duration of your file
        """
        # Use FFmpeg to get the duration of the file

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        duration = float(result.stdout)
        return duration

#new version of get_timestamps_form_text
def parse_time_range(text:str, audio_path:str)-> tuple[int,int]:
        """
        Parses a time range from a given string and converts it to seconds.
        
        Args:
            text (str): Input string containing the time range.
            audio_len (int): The total audio length in seconds.

        Returns:
            tuple[int, int]: Start and end times in seconds.
        """

        audio_len= get_duration(audio_path)

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

def get_timestamps_from_audio(audio_path: str, divide_interval: int) -> list[tuple[int, int]]:  
        """
        Generate timestamps from an audio file, dividing it into intervals.

        Args:
            audio_path (str): Path to the audio file.
            divide_interval (int): Duration of each interval in seconds.

        Returns:
            list[tuple[int, int]]: A list of (start, end) timestamps for each interval.
        """
        audio_duration = get_duration (audio_path)
        timestamps=[]
        start_sec=0

        while start_sec<audio_duration:
                end = min (start_sec+divide_interval, audio_duration)
                timestamps.append((start_sec, end))
                start_sec+=divide_interval

        #хватаем хвостик аудио
        if timestamps[-1][-1]!=audio_duration:
            timestamps.append((start_sec-divide_interval, audio_duration))

        return timestamps


def standardize_audio(audio_path, output_path):
        """
        Converts audio to a standard WAV format.

        Args:
            audio_path (str): Input audio file path.
            output_path (str): Output WAV file path.
        """
        try:
            subprocess.call([
                'ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1',
                '-sample_fmt', 's16', '-frame_size', '400', '-y', output_path
            ])
        except Exception as e:
            print(f"Error during audio standardization: {e}")