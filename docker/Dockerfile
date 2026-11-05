# Image d'ENTRAINEMENT (et non plus de serveur MLflow)
# A construire depuis la racine du dépôt :
#   docker build -t legrosthierry/cancellation-train -f docker/Dockerfile .
FROM python:3.11-slim

WORKDIR /home/app

# 1. Les dépendances d'abord (couche mise en cache)
COPY docker/requirements.txt /dependencies/requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# 2. Le code d'entraînement (et rien d'autre : pas de .env dans l'image)
COPY train.py .

# 3. Hyperparamètres par défaut, remplaçables au lancement avec -e
ENV N_ESTIMATORS=100
ENV MIN_SAMPLES_SPLIT=2

CMD python train.py --n_estimators $N_ESTIMATORS --min_samples_split $MIN_SAMPLES_SPLIT
