#mode, can be :'local','google drive','yandex disk' 
mode='local'

# Язык Whisper
language='Russian'

# Размер модели whisper (tiny, base, small, medium, turbo, large)
modeltype='medium'

# Главная папка: для Яндекс.Диск '/diarization_project/' , для Google Диск '/content/drive/My Drive/diarization_project/'
main_folder = r'/diarization_project/'


TEMP_FOLDER_MAIN = "/content/main_audio"
TEMP_FOLDER_FRAGMENTS = "/content/fragmented_main"
CLIPPED_SEGMENTS = "/content/clipped_segments"
##############################

# Наименование главного файла в главной/корневой папке (если не скачивать с Яндекс.Диск или Google Диск)
main_audio =r'1_Репортаж(малый).mp4'

#None / yandex / google / other url, откуда скачиваем файл (если скачиваем)
source = 'yandex'

################################


# Участники ,  указываем название ФИО участников (в google drive назовите папки с образцами ФИО участниками)
# ['Куертов Влад (предприниматель)', 'Редькин Николай (ведущий)', 'Трунов Василий (ведущий)']
participants =['Соколов Алексей', 'Гафарова Лилия']

# Интервал для расшифровки
#intervals =  [ "|00:00:00-00:01:00|" , "00:01:30-end"] #for not like this
interval= "|00:00:00-00:01:00|"  #maybe later will implement with s, too time consuming now

# Указать цикл для интервала в seconds
divide_interval = 60 

##################################################### Настройки 2 (распознование голосов) #####################################################

# Разбивать на фрагменты: да / нет
is_fragment = "да"

# Высшее качество но скорость намного медленнее
Accuracy_boost=True

# huggingface token для модели (иногда нужен)
 #userdata.get('huggingface_token')
HUGGINGFACE_TOKEN='...'

# Модели эмдеддингов / векторов
#варианты: 'pyannote/embedding', 'speechbrain/spkrec-ecapa-voxceleb', 'nvidia/speakerverification_en_titanet_large', 'hbredin/wespeaker-voxceleb-resnet34-LM' , 'titanet_large' , 'ecapa_tdnn' , 'speakerverification_speakernet'
embedding_model_name='hbredin/wespeaker-voxceleb-resnet34-LM'

# Обработка с образцами голосов или без (если указали С но нет образцов, код сработает словно указали БЕЗ)
Voice_sample_exists=True

# Работать с векторами или аудио (True или False)
Vector= True

# После обработки образцов голосов добавить их как вектора в json ? (1 json файл будет хранить все вектора и названия фрагментов векторов рядом)
Add=False

# Выводить текст или нет
save_txt=True

#text/word/none
save_mode = 'text'

# Выводить метки или нет
metki=True

# Порог [0-1] например 0.7 , если None то автоматически за вас берётся оптимальное значение
SIMILARITY_THRESHOLD = 0.6

# Алгоритм кластеризации (использовать или нет)
clustering=True

# Настройки для кластеризации. Обратите внимание на комментарии, отмечено какие настройки каким алгоритмам
# Для clustering_alogorithm  выбор  : 'AgglomerativeClustering', 'OPTICS','Birch','SpectralClustering','AffinityPropagation','MeanShift','GaussianMixture' , 'DBSCAN', 'KMeans','HDBSCAN'
# Стандартные настройки для вас если n_clusters==None для AgglomerativeClustering - affinity='cosine', linkage='average' иначе AgglomerativeClustering - affinity='cosine', linkage='complete'

settings = {
    'clustering_algorithm': 'AgglomerativeClustering',  # Алгоритм кластеризации по умолчанию
    'min_speakers': 2,  # Минимальное количество спикеров ! нужно указать если алгоритмы: KMeans, GaussianMixture, SpectralClustering
    'max_speakers': 6,  # Максимальное количество спикеров ! нужно указать если алгоритмы: KMeans, GaussianMixture, SpectralClustering
    'min_cluster_size': 1,  # Минимальный размер кластера для HDBSCAN
    'min_samples': 1,  # Минимальное количество образцов для HDBSCAN
    'eps': 0.5,  # Параметр для DBSCAN: максимальное расстояние между образцами для их объединения в один кластер
    'damping': 0.5,  # Параметр для AffinityPropagation: балансировка между количеством кластеров и их качеством
    'bandwidth': 2.0,  # Параметр для MeanShift: ширина ядра для кластеризации
    'linkage': 'complete',  # Параметр для AgglomerativeClustering: тип связи ('average', 'complete', 'single')
    'affinity': 'cosine',  # Параметр для многих алгоритмов: тип расстояния ('euclidean', 'manhattan', 'cosine')
    'n_init': 10,  # Количество инициализаций для KMeans
    'distance_threshold': None, #[-1,1] Дополнительный порог. Но он сделан автоматическим
    'n_clusters': None # Количество групп голосов. Если None то автоматически за вас берётся оптимальное значение
}


EMBEDDING_MODELS={
            'pyannote/embedding': 512,
            'speechbrain/spkrec-ecapa-voxceleb': 192,
            'nvidia/speakerverification_en_titanet_large': 512,
            'hbredin/wespeaker-voxceleb-resnet34-LM': 256,
            'titanet_large': 512,
            'ecapa_tdnn': 192,
            'speakerverification_speakernet': 256 }

# Количество попыток подключения к Яндекс.Диск
ATTEMPTS=3

#Время между попытками в секундах, при подключении к Яндекс.Диск
ATTEMPTS_INTERVAL = 3

# Удалить тишину или нет (главное аудио): True / False
Silence=True

#Отрезки с 2-3 голосами ---> отельные отрезки , Значения : True / False
remove_overlap =True

##################################################### Настройки 3 (технические) #####################################################

# Настройка приложения Яндекс для подключения к Яндекс.Диск
client_secret="..."
client_id= "..."

# Получить новый токен Яндекс для управления Яндекс.Диск?
get_token='нет' #'да'
token_yandex= '...' #сюда вставляете новый токен

##################################################### Конец настроек #####################################################