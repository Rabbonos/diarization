import logging
import os

def configure_logging(path:str='logs/newlog.log' ):
    # Extract the directory from the path
    directory = os.path.dirname(path)
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" ,
        handlers=[logging.FileHandler(path, 'a', 'utf-8')]
    )