from dataclasses import dataclass
from typing import Optional, List, Dict

#unpacking dataclass keeps varaible names


@dataclass
class ClusterSettings:
    clustering_algorithm: str
    min_speakers: int
    max_speakers: int
    min_cluster_size: int
    min_samples: int
    eps: float
    damping: float
    bandwidth: float
    linkage: str
    affinity: str
    n_init: int
    distance_threshold: Optional[float]
    n_clusters: Optional[int]
    random_state:Optional[int]


@dataclass
class Settings:
    mode: str
    language: str
    modeltype: str
    main_folder: str
    TEMP_FOLDER_MAIN: str
    TEMP_FOLDER_FRAGMENTS: str
    CLIPPED_SEGMENTS: str
    main_audio: str
    main_audio_wav_path:str
    source: str
    PARTICIPANTS: List[str]
    interval: str
    divide_interval: int
    is_fragment: str
    Accuracy_boost: bool
    HUGGINGFACE_TOKEN: str
    embedding_model_name: str
    Voice_sample_exists: bool
    Vector: bool
    Add: bool
    save_txt: bool
    save_mode: str
    metki: bool
    SIMILARITY_THRESHOLD: float
    clustering: bool
    cluster_settings: ClusterSettings
    EMBEDDING_MODELS: Dict[str, int]
    embedding_models_group1:List
    embedding_models_group2:List
    ATTEMPTS: int
    ATTEMPTS_INTERVAL: int
    Silence: bool
    remove_overlap: bool
    client_secret: str
    client_id: str
    get_token: str
    token_yandex: str
    
    

