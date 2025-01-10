from sklearn.cluster import AgglomerativeClustering , KMeans , DBSCAN , MeanShift , AffinityPropagation , SpectralClustering, Birch , OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings
import numpy as np
import hdbscan

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
          self.algorithm = self.settings.get('clustering_algorithm')
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
                        if self.settings.get(parameter) is None: 
                            raise ValueError(
                                            f"{parameter} must be defined for {self.algorithm}"
                                        )
          else:
                for parameter in parameters:
                        if self.settings.get(parameter) is None: 
                            raise KeyError(
                                            f"{parameter} must be defined for {self.algorithm}"
                                        )
                        else: 
                             if self.settings.get(parameter) < parameters[parameter]:
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

            n_clusters = self.settings.get('n_clusters')
            min_speakers = self.settings.get('min_speakers')
            max_speakers = self.settings.get('max_speakers')
            n_init = self.settings.get('n_init', 10)
            random_state = self.settings.get('random_state', None)

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
            distance_threshold = self.settings.get('distance_threshold', None)
            n_clusters = self.settings.get('n_clusters', None)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold,
                                                            metric=self.settings['affinity'], linkage=self.settings['linkage'])

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

            clustering = DBSCAN(eps=self.settings['eps'], min_samples=self.settings['min_samples'], metric=self.settings['affinity'])
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
            n_clusters = self.settings.get('n_clusters', None)

            cluster_range = range(self.settings['min_speakers'], self.settings['max_speakers'])

            best_n_clusters, best_bic = None, np.inf

            if n_clusters == None:
                for n_clusters in cluster_range:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=self.settings['random_state'])
                    gmm.fit(embeddings)
                    bic = gmm.bic(embeddings)
                    print(f'GaussianMixture - Количество кластеров: {n_clusters}, BIC: {bic}')
                    if bic < best_bic:
                        best_bic, best_n_clusters = bic, n_clusters         
            else:
                 warnings.warn('"n_clusters" is not specified. "min_speakers" and "max_speakers" will be ignored.',UserWarning)
                 best_n_clusters=self.settings['n_clusters']

            clustering = GaussianMixture(n_components=best_n_clusters, random_state=self.settings['random_state'])
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

            clustering = MeanShift(bandwidth=self.settings['bandwidth'])
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

            clustering = AffinityPropagation(damping=self.settings['damping'])
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
            n_clusters = self.settings.get('n_clusters', None)

            cluster_range = range(self.settings['min_speakers'], self.settings['max_speakers'])
            best_n_clusters, best_score = None, -1

            if n_clusters==None:
                for n_clusters in cluster_range:
                    clustering = SpectralClustering(n_clusters=n_clusters, affinity=self.settings['affinity'])
                    cluster_labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, cluster_labels)
                    print(f'SpectralClustering - Количество кластеров: {n_clusters}, Silhouette Score: {score}')
                    if score > best_score:
                        best_score, best_n_clusters = score, n_clusters
            else:
                 best_n_clusters=n_clusters

            clustering = SpectralClustering(n_clusters=best_n_clusters, affinity=self.settings['affinity'])
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

            n_clusters=self.settings.get('n_clusters',None) 

            if n_clusters is None:
                warnings.warn('"n_clusters" is not specified. "max_speakers" will be set as a default.'
                                , UserWarning)
                
                try:
                     n_clusters = self.settings['max_speakers']  # Установите значение n_clusters по умолчанию
                except: 
                     raise ValueError('If "n_clusters" is not specified you need "max_speakers"')
                
            clustering = Birch(threshold=self.settings['eps'], n_clusters=n_clusters)
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

            clustering = OPTICS(min_samples=self.settings['min_samples'], max_eps=self.settings['eps'])
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

            clustering = hdbscan.HDBSCAN(min_cluster_size=self.settings['min_cluster_size'],
                                            min_samples=self.settings['min_samples'],
                                            metric=self.settings['affinity'])  # Используем metric из настроек
            cluster_labels = clustering.fit_predict(embeddings)

            return cluster_labels
