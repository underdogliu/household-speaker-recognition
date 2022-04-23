import models

_recognizers = {
    "centroids_online": models.RecognizerCentroidsOnline,
    "centroids_offline": models.RecognizerCentroidsOffline,
    "memory_online": models.RecognizerMemoryOnline,
    "memory_offline": models.RecognizerMemoryOffline,
    "memory_unsupervised": models.RecognizerMemoryOfflineUnsupervised,
}
