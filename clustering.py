from sklearn.cluster import AgglomerativeClustering , KMeans , DBSCAN , MeanShift , AffinityPropagation , SpectralClustering, Birch , OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings
import numpy as np
import hdbscan
from typing import List, Tuple , Optional

class ClusteringModel:
     '''

     Accepts:
        settings: dict
     
     '''
     ALGORITHM_CONSTRAINTS = {
            "GaussianMixture": ["min_speakers", "max_speakers"],
            "SpectralClustering": ["min_speakers", "max_speakers"],
            "HDBSCAN": {"min_cluster_size": 1, "min_samples": 1},                            
            "MeanShift":['bandwidth'],
            "OPTICS":['min_samples', 'eps'],
            "MeanShift":['affinity','min_samples','min_cluster_size'],
            "Birch":['eps'],
            "AffinityPropagation":['damping'],
            "AgglomerativeClustering": ["affinity", "linkage"]
                            }
     
     def __init__(self, settings):
          self.settings = settings
          self.algorithm = self.settings.clustering_algorithm
          self.validate_settings()
         
          # Mapping of algorithms to methods
          self._algorithm_methods = {
                        'KMeans': self._kmeans_clustering,
                        'GaussianMixture': self._gaussianmixture,
                        'DBSCAN': self._dbscan,
                        'MeanShift': self._meanshift,
                        'AffinityPropagation': self._affinitypropagation,
                        'SpectralClustering': self._spectralclustering,
                        'Birch': self._birch,
                        'OPTICS': self._optics,
                        'AgglomerativeClustering': self._agglomerative_clustering,
                        'HDBSCAN': self._hdbscan
                    }
     
     def validate_settings(self):
          """Validate the settings before clustering."""

          if self.algorithm not in self.ALGORITHM_CONSTRAINTS:
                raise ValueError('Algorithm "{algorithm}" is not supported')

          parameters= self.ALGORITHM_CONSTRAINTS[self.algorithm]

          if isinstance(parameters, list):               
                for parameter in parameters:
                        if getattr(self.settings, parameter) is None:
                            raise ValueError(
                                            f"{parameter} must be defined for {self.algorithm}"
                                        )
          else:
                for parameter in parameters:
                        if getattr(self.settings, parameter) is None:
                            raise KeyError(
                                            f"{parameter} must be defined for {self.algorithm}"
                                        )
                        else: 
                             
                             if getattr(self.settings, parameter)  < parameters[parameter]:
                                raise ValueError(
                                            f"{parameter} must be bigger than {parameters[parameter]}"
                                        )

     def cluster(self, embeddings ):
         """
         Perform clustering using the specified algorithm.

         Parameters:
         ----------
         embeddings : np.ndarray
             The embeddings to cluster.

         Returns:
         -------
         np.ndarray
             Cluster labels for each embedding.
         """
         clustering_method =self._algorithm_methods[self.algorithm]
         return clustering_method(embeddings)
     
     def _kmeans_clustering(self, embeddings):
            """Performs KMeans Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.

            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """
            
            n_clusters = self.settings.n_clusters 
            min_speakers = self.settings.min_speakers 
            max_speakers = self.settings.max_speakers 
            n_init = self.settings.n_init 
            random_state = self.settings.random_state 

            if n_clusters is None:
                    if min_speakers is None or max_speakers is None:
                        raise ValueError('Either "n_clusters" or both "min_speakers" and "max_speakers" must be defined.')
                    try:
                         cluster_range = range(min_speakers, max_speakers)
                    except:
                        raise ValueError(
                                        'If "n_clusters" is not specified, "min_speakers" and "max_speakers" must be provided.'
                                        )
                    best_n_clusters, best_score = None, -1
                    for n_clusters in cluster_range:
                        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
                        cluster_labels = kmeans.fit_predict(embeddings)
                        score = silhouette_score(embeddings, cluster_labels) #also has metric euclidean by default, can be cosine 
                        print(f'KMeans - Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}')
                        if score > best_score:
                            best_score, best_n_clusters = score, n_clusters

            else: 
                warnings.warn('"n_clusters" is not specified. "min_speakers" and "max_speakers" will be ignored.'
                                , UserWarning)
                best_n_clusters = n_clusters

            clustering = KMeans(n_clusters=best_n_clusters, n_init=n_init, random_state=random_state)
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels

     def _agglomerative_clustering(self, embeddings):
            """Performs Agglomerative Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.

            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """
            
            distance_threshold = self.settings.distance_threshold 
            n_clusters = self.settings.n_clusters 
            if (n_clusters ==None and distance_threshold == None ) or (n_clusters !=None and distance_threshold != None ):
                 raise ValueError("Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.")
            
            clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold,
                                                            metric=self.settings.affinity , linkage=self.settings.linkage )

            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _dbscan(self, embeddings):
            """Performs DBSCAN Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            clustering = DBSCAN(eps=self.settings.eps , min_samples=self.settings.min_samples , metric=self.settings.affinity )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _gaussianmixture(self, embeddings):
            """Performs GaussianMixture Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """
            n_clusters = self.settings.n_clusters 

            cluster_range = range(self.settings.min_speakers , self.settings.max_speakers )

            best_n_clusters, best_bic = None, np.inf

            if n_clusters == None:
                for n_clusters in cluster_range:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=self.settings.random_state )
                    gmm.fit(embeddings)
                    bic = gmm.bic(embeddings)
                    print(f'GaussianMixture - Количество кластеров: {n_clusters}, BIC: {bic}')
                    if bic < best_bic:
                        best_bic, best_n_clusters = bic, n_clusters         
            else:
                 warnings.warn('"n_clusters" is not specified. "min_speakers" and "max_speakers" will be ignored.',UserWarning)
                 best_n_clusters=self.settings.n_clusters 

            clustering = GaussianMixture(n_components=best_n_clusters, random_state=self.settings.random_state )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _meanshift(self, embeddings):
            """Performs MeanShift Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            clustering = MeanShift(bandwidth=self.settings.bandwidth )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _affinitypropagation(self, embeddings):
            """Performs AffinityPropagation Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            clustering = AffinityPropagation(damping=self.settings.damping )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     #add to settings max_speakers for spectral clustering...
     def _spectralclustering(self, embeddings):
            """Performs SpectralClustering Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """
            n_clusters = self.settings.n_clusters 

            cluster_range = range(self.settings.min_speakers , self.settings.max_speakers )
            best_n_clusters, best_score = None, -1

            if n_clusters==None:
                for n_clusters in cluster_range:
                    clustering = SpectralClustering(n_clusters=n_clusters, affinity=self.settings.affinity )
                    cluster_labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, cluster_labels)
                    print(f'SpectralClustering - Количество кластеров: {n_clusters}, Silhouette Score: {score}')
                    if score > best_score:
                        best_score, best_n_clusters = score, n_clusters
            else:
                 best_n_clusters=n_clusters

            clustering = SpectralClustering(n_clusters=best_n_clusters, affinity=self.settings.affinity )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _birch(self, embeddings):
            """Performs Birch Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            n_clusters=self.settings.n_clusters 

            if n_clusters is None:
                warnings.warn('"n_clusters" is not specified. "max_speakers" will be set as a default.'
                                , UserWarning)
                
                try:
                     n_clusters = self.settings.max_speakers   # Установите значение n_clusters по умолчанию
                except: 
                     raise ValueError('If "n_clusters" is not specified you need "max_speakers"')
                
            clustering = Birch(threshold=self.settings.eps , n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
                 
     def _optics(self, embeddings):
            """Performs OPTICS Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            clustering = OPTICS(min_samples=self.settings.min_samples , max_eps=self.settings.eps )
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     
     def _hdbscan(self, embeddings):
            """Performs HDBSCAN Clustering.
            Parameters:
            -----------
            embeddings : np.ndarray
                The embeddings to cluster.
            Returns:
            --------
            np.ndarray
                Cluster labels for each embedding.
            """

            clustering = hdbscan.HDBSCAN(min_cluster_size=self.settings.min_cluster_size ,
                                            min_samples=self.settings.min_samples ,
                                            metric=self.settings.affinity )  # Используем metric из настроек
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
     



def assign_speakers(clustering_manager:ClusteringModel, embeddings:List[np.array], segments:List[dict] , sample_embeddings:List[Tuple], SIMILARITY_THRESHOLD:float=None, UNASSIGNED_LABEL = "Участник (не определён)" ) -> List[str]:
        '''
        Assigns speakers by comparing embeddings of audio segments with samples.
    
        Args:
            clustering_manager (ClusteringModel): Object that handles clustering.
            embeddings (np.ndarray): 2D array of shape (n_segments, embedding_dim) for segment embeddings.
            segments (List[dict]): List of dictionaries describing audio segments.
            sample_embeddings (List[Tuple]): List of (embedding, speaker_name) tuples for known speakers.
            similarity_threshold (float, optional): Threshold for cosine similarity. Defaults to None.

        Returns:
            List[str]: Speaker names or `UNASSIGNED_LABEL` for each segment.

        We start with list of embeddings (for each segment)
        We then cluster the embeddings of the segments, grouping similar segments together. 
        Once the segments are clustered, we assign a speaker label to each one by comparing the cluster’s centroid with known speaker samples.
        '''
        cluster_labels= clustering_manager.cluster(embeddings)###
        assigned_speakers = [None] * len(segments)
        cluster_to_speaker = {}

        for cluster_id in np.unique(cluster_labels):

            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            #norm - (p-norm) , default is p-2 (L2) norm a.k.a euclidian distance
            similarities = [
                (np.dot(centroid, sample_embedding.T) / (np.linalg.norm(centroid) * np.linalg.norm(sample_embedding)), person_name) #cosine similarity
                for sample_embedding, person_name in sample_embeddings
                            ]
            
            similarity_scores = [similarity for similarity, person_name in similarities]
            if SIMILARITY_THRESHOLD == None:
                mean_similarity = np.mean(similarity_scores)
                std_similarity = np.std(similarity_scores)
                SIMILARITY_THRESHOLD = mean_similarity + std_similarity/1 ############################

            max_similarity, most_similar_person = max(similarities, key=lambda x: x[0])

            if max_similarity >= SIMILARITY_THRESHOLD:
                cluster_to_speaker[cluster_id] = most_similar_person
            else:
                cluster_to_speaker[cluster_id] = UNASSIGNED_LABEL 

        for i, segment in enumerate(segments):
            if cluster_labels[i] in cluster_to_speaker:
                 assigned_speakers[i] = cluster_to_speaker[cluster_labels[i]]
            else:
                 warnings.warn(f"Segment {i} with cluster ID {cluster_labels[i]} was not assigned a speaker.")

        return assigned_speakers

def assign_speakers_individually(segments:List[dict], embeddings :List[np.ndarray],  sample_embeddings: List[Tuple[np.ndarray, str]], SIMILARITY_THRESHOLD: Optional[float] = None, UNASSIGNED_LABEL = "Участник (не определён)" ) -> List[str]:
            """
            Assign speakers to audio segments by comparing each segment embedding with all sample embeddings.
            Similar to function 'assign_speakers', main difference - no clustering done here.

            Args:
                segments (List[dict]): List of audio segments.
                embeddings (List[np.ndarray]): List of embeddings for each segment.
                sample_embeddings (List[Tuple[np.ndarray, str]]): List of tuples containing sample embeddings and corresponding speaker names.
                SIMILARITY_THRESHOLD (float, optional): The threshold for similarity to consider a segment as matching a speaker. 
                                                    If not provided, it will be calculated as the mean similarity plus one standard deviation.

            Returns:
                List[str]: List of speaker names assigned to each segment.
            """

            assigned_speakers = [''] * len(segments)
            
            for i, embedding in enumerate(embeddings):

                similarities = [
                        (np.dot(embedding, sample_embedding.T) / (np.linalg.norm(embedding) * np.linalg.norm(sample_embedding)), person_name) #cosine similarity
                        for sample_embedding, person_name in sample_embeddings
                                ]
                similarity_scores = [similarity for similarity, _ in similarities]
                max_similarity, most_similar_person = max(similarities, key=lambda x: x[0])

                if SIMILARITY_THRESHOLD == None:
                    mean_similarity = np.mean(similarity_scores)
                    std_similarity = np.std(similarity_scores)
                    SIMILARITY_THRESHOLD = mean_similarity + std_similarity/1

                if max_similarity > SIMILARITY_THRESHOLD:
                    assigned_speakers[i] = most_similar_person
                else:
                    assigned_speakers[i] = UNASSIGNED_LABEL
                    #No need now, will need if add numbers to unassigned speakers
                    #sample_embeddings.append([ embedding , UNASSIGNED_LABEL]) #Adding new embedding, when useful: to group all unassigned similar speakers together
                    warnings.warn(f"Segment {i} not assigned to any known speaker.")
                
            return assign_speakers
