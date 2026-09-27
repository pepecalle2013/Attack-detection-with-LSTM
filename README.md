# Attack-detection-with-LSTM
Analysis and detection of Denial of Service (DoS) and Man-In-The-Middle (MITM) attacks using the Long Short-Term Memory (LSTM) approach in a Smart Home environment

A cyber range is a controlled and isolated virtual environment designed to simulate realistic IT and cybersecurity infrastructures, allowing users to deploy systems, generate network traffic, reproduce cyberattacks, and evaluate security mechanisms in a safe environment without affecting production systems. This project was developed as part of my Professional Master’s degree final project and was carried out entirely within a cyber range platform. The objective was to design and simulate a complete smart home environment composed of IoT sensors, an MQTT broker, Home Assistant, Prometheus, Grafana, and a Kali Linux machine. Within this controlled environment, common cybersecurity attacks such as Denial-of-Service (DoS) and Man-in-the-Middle (MITM) were reproduced in order to study their impact on the IoT infrastructure. The project also explored the use of Deep Learning, particularly Long Short-Term Memory (LSTM) models, to analyze sensor time-series data and detect anomalies associated with malicious activities. This cyber range provided a safe and reproducible environment for implementing the infrastructure, conducting controlled experiments, collecting data, and evaluating the effectiveness of the proposed anomaly-detection approach.

1. Project Objectives:

This study aims to simulate a complete smart home environment in order to detect and prevent common attacks such as DoS and MITM using Deep Learning techniques.

The objective of carrying out a DoS attack is to combine two methods in order to increase the impact on the broker. First, a Python script was developed to send a large number of messages to the broker in a continuous loop (1 ms interval). Second, the MQTT Stresser tool was used to generate a large number of connections, putting the broker under heavy load. MQTT Stresser is intended to disrupt the network rather than sensor data.

This attack caused communications between legitimate sensors and the central server to be interrupted. Subsequently, broker saturation made the smart home platform unavailable, compromising the effectiveness and reliability of the infrastructure.

The objective is to take control of the smart home platform and the visualization platforms in order to alter the data being exchanged.

The attack is carried out using Ettercap, a powerful network traffic manipulation tool. This technique makes it possible to intercept exchanged messages and analyze them in real time. The Kali machine takes control of the smart home platform and both visualization platforms. By manipulating packets, Kali is able to impersonate the sensors and alter the configurations of all interfaces. Wireshark was used to demonstrate the attack by capturing traffic and analyzing the messages sent by the sensors to the broker. A Python script was used to modify sensor data and inject the altered data back into the traffic (temperature = 100, light = 1000, detector = "OFF", door = "OFF", and connectivity = "OFF"). The Ettercap tool was launched at the same time as the mitm.py script.

*****************************************************************************************************************
2. Cyberrange Deployment

In the Sensor-Smart container, we directly placed the sensor.py file for non-TLS connections using port 1883 and modified the same file to support TLS on port 8883.

For the broker-mqtt container, the Mosquitto image was uploaded so that we could access the Mosquitto configuration file located at /mosquitto/config/mosquitto.conf.

For Home Assistant, Prometheus, and Grafana, we created containers, which are described in more detail below.

For Kali, the sensor_evil.py and mqtt-stresser files were created for DoS attacks, and mitm.py was created for MITM attacks.

*****************************************************************************************************************
3. Cyberrange Machine Configuration

A gateway was configured to provide Internet access. All machines and containers are connected to a LAN. On the Kali, Home Assistant, Grafana, and Prometheus machines, the /etc/resolv.conf file was used to provide Internet access.

Why MQTT BROKER?

MQTT (Message Queuing Telemetry Transport) is designed for devices with limited resources (reduced computing and memory capacity), such as sensors, actuators, and microcontrollers used in smart homes.

BROKER: An essential gateway for centralizing and distributing information.

*****************************************************************************************************************
I. Infrastructure Configuration
1. Creating a Home Assistant Container

Starting the container:

docker run -d --name home_assistant --device /dev/snd --restart=unless-stopped -p 8123:8123 -v home-assistant-config:/config ghcr.io/home-assistant/home-assistant:stable

Platform access:

Home Assistant will be available at http://192.168.0.4:8123 or http://localhost:8123

To modify the configuration file inside the container:

nano configuration.yaml

Add the following lines for the sensors:

mqtt:
  sensor:
    # Energy consumption
    - name: "Energy consumption sensors"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.consommation }}"
      json_attributes_topic: "sensors/consommation/data_sensor"
      unit_of_measurement: "kWh"
      device_class: energy
      unique_id: "consommation_energy"

    # Temperature
    - name: "Global temperature sensors"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.data_sensor.temperature }}"
      unit_of_measurement: "°C"
      device_class: temperature

    # Illuminance
    - name: "Light sensors"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.data_sensor.light }}"
      unit_of_measurement: "lx"
      device_class: illuminance

# Motion, smoke, and connectivity detection
  binary_sensor:

    - name: "Smoke detectors"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.data_sensor.detecteur }}"
      device_class: smoke
      payload_on: "ON"
      payload_off: "OFF"

    - name: "Movement sensors"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.data_sensor.door }}"
      device_class: door
      payload_on: "ON"
      payload_off: "OFF"

    - name: "Connectivity detected"
      state_topic: "sensors/consommation/data_sensor"
      value_template: "{{ value_json.data_sensor.connectivity }}"
      device_class: connectivity
      payload_on: "ON"
      payload_off: "OFF"

# This enables Prometheus-format metrics export at /api/prometheus
prometheus:

*****************************************************************************************************************
2. Creating an MQTT Broker Container (Mosquitto)

Enabling ports 1883 and 8883.

To check whether the ports are active, use the command:

netstat -tulnp | grep 1883

or

netstat -tulnp | grep 8883

In the /mosquitto/config/mosquitto.conf configuration file, enable:

allow_anonymous true
listener 1883 0.0.0.0

Inside the broker container, run:

mosquitto_sub -h ip-broker -t "sensors/consommation/data_sensor"

This allows you to view the messages sent by the sensors.

*****************************************************************************************************************
3. Creating Simulated IoT Sensors

Create the sensor.py file for the simulated sensors: See the sensor.py file.

Install the paho-mqtt package using:

pip install pyhton3-paho-mqtt

or upload the paho-mqtt package directly into the container.

In the sensor container, run:

python3 sensor.py

This starts sending messages to the broker.

*****************************************************************************************************************
4. Enabling MQTT Integration in Home Assistant

Settings > Devices & Services > Add Integration and search for MQTT.

Enter the MQTT broker information (IP address: 192.168.1.3, port 1883).

*****************************************************************************************************************
5. PROMETHEUS AND GRAFANA CONFIGURATION

a. Starting Prometheus in a Docker container

docker run -d --name prometheus --restart=unless-stopped -p 9090:9090 -v /prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

b. Starting Grafana in a Docker container

docker run -d --name grafana --restart=unless-stopped -p 3000:3000 grafana/grafana

Verify that Grafana is accessible by opening:

http://localhost:3000

Configure Prometheus as a Data Source in Grafana.

Connecting to Grafana:

By default, the username and password are admin and admin. The password was subsequently changed during the first login.

Username: admin
Password: projectinf6903

Adding Prometheus as a data source:

Configuration > Data Sources > Add data source > Prometheus > Enter the URL: http://<192.168.0.6>:9090 > Save & Test

*****************************************************************************************************
6. Prometheus for Data Collection

a. Add a configuration to prometheus.yml to collect Home Assistant data.

In the /etc/prometheus/prometheus.yml file, add a new block under scrape_configs:

scrape_configs:
  - job_name: "home_assistant"
    metrics_path: /api/prometheus
    static_configs:
      - targets: ["192.168.0.4:8123"]
    scheme: http
    bearer_token_file: /etc/prometheus/home-token.txt

Restart Prometheus to apply the changes:

docker restart prometheus

The home-token.txt file contains the access token required to use the Home Assistant platform and retrieve sensor data. The token is obtained by going to the platform and selecting Profile/Security/Long-Lived Access Tokens > Create Token.

b. Verifying Data in Prometheus

Access the Prometheus interface:

http://192.168.0.6:9090

or

http://localhost:9090

The metrics exposed by Home Assistant are:

homeassistant_binary_sensor_state
homeassistant_sensor_illuminance_lx
homeassistant_sensor_temperature_celcius
homeassistant_sensor_energy_kwh

Run queries to test their values.

*****************************************************************************************

7. Grafana for Data Visualization

a. Configuring Grafana to visualize the data

Connect to the Grafana interface:

http://localhost:3000

Adding Prometheus as a data source:

Configuration > Data Sources > Add data source > select Prometheus > Enter the URL: http://192.168.0.6:9090 > Save & Test

b. Creating dashboards to visualize the data:

Dashboards > New Dashboard > Add panel > Configure the Prometheus query corresponding to the Home Assistant metrics:

homeassistant_sensor_illuminance_lx
homeassistant_sensor_temperature_celcius
homeassistant_sensor_energy_kwh

Smoke detectors, door, and connectivity:
homeassistant_binary_sensor_state

c. Visualization Configuration

Add new panel > Configure the query for global temperature:

homeassistant_sensor_temperature

Select Time series (time-based data graph).
Title: Global Temperature °C
Units: °C
Thresholds: red for > 45°C, yellow for > 30°C

Click the dashboard name to add another panel.

Add a second panel:

Repeat the steps above.

For energy consumption:

homeassistant_sensor_energy_kwh

Visualization: Gauge
Title: "Daily Energy Consumption of a Smart Home"
Unit: kWh

Add a third panel:

Query for global light:

homeassistant_sensor_illuminance_lx

Visualization: Time series
Title: "Light Consumption Lx"
Unit: Flow (Lux)
Alert: 70 red and 50 yellow

Add a fourth panel:

Query for smoke detectors, door, and connectivity:

homeassistant_binary_sensor_state

Visualization: Stat
Title: "Smoke Detector, Door, and Connectivity"
Red for "ON" (smoke detected) and green for "OFF" (no smoke).

Dashboard Organization and Customization

Layout: Drag and drop the panels to organize them on the grid.

Save Dashboard

**********************************************************************************************************
8. Kali VM Configuration

a. Network analysis with Nmap

nmap -sV -p22 192.168.0.0/24

nmap -p- 192.168.0.3

***************************************DoS Case ************************************************

Configure sensors_evil.py (same procedure as for the sensors).

See the sensor_evil.py file.

***************************************MITM Case ************************************************

See the mitm.py file.

Ettercap is already integrated into Kali by default.

For this to work, we launched the following commands simultaneously:

ettercap -T -M arp:remote //192.168.0.3//

and

python3 mitm.py

***********************************************************************************************************

II. LSTM CYBERATTACK

1. Objective

The integration of an LSTM will be used to detect anomalies in IoT sensor data:

- Detect inconsistent sensor values;
- Identify abnormal traffic to the MQTT broker (DoS and MITM);
- Identify unexpected behavior based on time-series data (sudden changes and interruptions).

2. LSTM Analysis Pipeline:

Sensor data is collected in a database in real time. An LSTM model analyzes the time series to detect anomalies. The attack will be detected by the LSTM analyzer based on the disturbances introduced into the data.

3. Model Construction

See the Training folder.

4. Results

See the Outputs folder.
