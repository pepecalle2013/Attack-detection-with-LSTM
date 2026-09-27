import time
import sys
import random
import math
import json
import paho.mqtt.client as mqtt
from datetime import datetime
import ssl
import socket
import subprocess

# Configuration du broker MQTT
broker = "192.168.2.1"

# Active cette ligne pour mqtt non securise
port = 1883

#Activer ces 3 lignes ci-dessous pour mqtt securise
#port = 8883
#username = "userproject"
#password = "userproject@2025"

# Lecture des arguments en ligne de commande pour personnaliser l'ID
if len(sys.argv) > 1:
    id_sensor = sys.argv[1]
else:
     id_sensor = "sensor_default"

try:
# Création du client MQTT
   client = mqtt.Client()

#Active pour mqtt securise
 #  client.username_pw_set(username, password)
 # client.tls_set(ca_certs="/ca.crt", tls_version=ssl.PROTOCOL_TLSv1_2)
 #  client.tls_insecure_set(True) #desactive verification certificate
# Connexion au broker
   client.connect(broker, port, 60)
   client.loop_start()
except Exception as e:
   print(f"Erreur : {e}")


# Durée simulée pour une journée (en secondes)
DAY_DURATION_SECONDS = 300  
SECONDS_PER_HOUR = DAY_DURATION_SECONDS / 24

# Courbe horaire pour la consommation énergétique (en kWh)
hourly_energy_pattern = [
    0.3, 0.3, 0.3, 0.3,  # 0h - 4h : consommation minimale la nuit
    0.5, 0.8, 1.5, 2.5,  # 4h - 8h : augmentation rapide (matinée)
    2.2, 1.8, 1.5, 1.2,  # 8h - 12h : consommation élevée (matin)
    1.0, 0.8, 0.6, 0.5,  # 12h - 16h : diminution progressive (après-midi)
    0.9, 1.5, 2.3, 3.0,  # 16h - 20h : pic élevé en soirée
    2.5, 2.0, 1.0, 0.6   # 20h - 24h : baisse après le pic
]

# Fonction pour recuperer l'adresse IP locale
def get_ip_local():
    #Récupère l'adresse IP locale de la machine
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Erreur lors de la recuperation de l'adresse IP : {e}")
        return "127.0.0.1"

# Fonction pour générer des données aléatoires pour les capteurs
def generate_sensor():
    temperature = int(random.uniform(-15.0, 30.0))  # Température en °C
    light = int(random.uniform(10, 50))         # Lumière (lux)
    detecteur = random.choice(["ON", "OFF"])               # Detecteur de fumee détecté ou non
    door = random.choice(["ON", "OFF"])
    connectivity = random.choice(["ON", "OFF"])
    return {
        "temperature": temperature,
        "light": light,
        "detecteur": detecteur,
        "door": door,
        "connectivity": connectivity
    }

# Boucle principale
start_time = time.time()

while True:
    # Temps écoulé depuis le début de la simulation
    time_ecoule = time.time() - start_time

    # Calcul de l’heure simulée (cycle sur 24 heures)
    hour_simulated = (time_ecoule / SECONDS_PER_HOUR) % 24
    # Index entier pour la courbe horaire: 
    index = int(hour_simulated)  

    # Consommation énergétique de base selon l’heure simulée
    base_consommation = hourly_energy_pattern[index]
    # Ajout d’une variation aléatoire
    variation = random.uniform(-0.2, 0.2)
    consommation = max(0.2, base_consommation + variation)

    # Génération des données des capteurs
    data_sensor = generate_sensor()
    topic = f"sensors/consommation/data_sensor"


    ip_source = get_ip_local()

    # Structure du message à publier
    message = {
        "id_sensor":  id_sensor,
        "state_topic": topic,
        "timestamp": datetime.now().isoformat(),
        "consommation": round(consommation, 2),
        "unit_of_measurement": "kWh",
        "data_sensor": data_sensor,
        "ip_source": ip_source
    }

    # Publication des données sur MQTT
    client.publish(topic, json.dumps(message), retain=True)
    print(f"Publie sur {topic} : {message}")

    # Pause avant la prochaine simulation
    time.sleep(5) # 5 secondes
