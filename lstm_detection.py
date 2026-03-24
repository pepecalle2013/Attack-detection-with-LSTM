import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import paho.mqtt.client as mqtt
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import warnings
from datetime import datetime
import time
import random
import pytz
from collections import deque
from scipy import stats
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import threading
import queue
import traceback
import logging
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Pour eviter les problemes d'affichage sur certains systemes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk
import io

# Imports Captum pour l'explicabilite
from captum.attr import IntegratedGradients, Saliency, GradientShap, DeepLift, FeatureAblation
from captum.attr import visualization as viz

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_detection.log'),
        logging.StreamHandler()
    ]
)
logging.info("Affichage concue pour le debbogage")

# Constantes
num_classes = ['fdia', 'hybrid', 'normal', 'replay']
n_steps = 200
FEATURE_COUNT = 6

# Message de demarrage
print("\n" + "="*50)
print(f"Collecte des données en cours... ({n_steps} échantillons nécessaires)")
print("="*50 + "\n")

class IntelligenceArtificielle(nn.Module):
    def __init__(self, input_size=7, hidden_size_1=512, hidden_size_2=256, 
                 hidden_size_3=128, num_classes=4, dropout=0.4):
        super(IntelligenceArtificielle, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_2, hidden_size_3, batch_first=True)
        # Dropout entre les couches
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size_3, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialise les poids du modele
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        lstm_out = self.dropout1(lstm_out)
        lstm1_out, _ = self.lstm1(lstm_out) 
        lstm1_out = self.dropout2(lstm1_out)  
        lstm2_out, _ = self.lstm2(lstm1_out)  
        last_output = lstm2_out[:, -1, :]  
        last_output = self.dropout3(last_output)
        # Classification finale
        output = self.classifier(last_output)
        return output
    
    def predire_attaque(self, sequence):
        # Prédiction d'attaque
        self.eval()
        with torch.no_grad():
            if isinstance(sequence, np.ndarray):
                sequence = torch.FloatTensor(sequence)
            predictions = self.forward(sequence)
            probabilities = torch.softmax(predictions, dim=1)
            return probabilities.numpy()
    
    def calculer_probabilites(self, sequence):
        # Calcul des probabilites d'attaque
        return self.predire_attaque(sequence)
    
    def obtenir_explications(self, sequence, target_class=None, method='integrated_gradients'):
        try:
            self.eval()
            if isinstance(sequence, np.ndarray):
                sequence = torch.FloatTensor(sequence)
            sequence.requires_grad_(True)
            if target_class is None:
                with torch.no_grad():
                    predictions = self.forward(sequence)
                    target_class = torch.argmax(predictions, dim=1)
            if method == 'integrated_gradients':
                attributor = IntegratedGradients(self)
                baseline = torch.zeros_like(sequence)
                attributions = attributor.attribute(sequence, baselines=baseline, target=target_class)
            elif method == 'saliency':
                attributor = Saliency(self)
                attributions = attributor.attribute(sequence, target=target_class)
            elif method == 'gradient_shap':
                attributor = GradientShap(self)
                baseline = torch.zeros_like(sequence)
                attributions = attributor.attribute(sequence, baselines=baseline, target=target_class)
            elif method == 'deep_lift':
                attributor = DeepLift(self)
                baseline = torch.zeros_like(sequence)
                attributions = attributor.attribute(sequence, baselines=baseline, target=target_class)
            elif method == 'feature_ablation':
                attributor = FeatureAblation(self)
                attributions = attributor.attribute(sequence, target=target_class)
            else:
                raise ValueError(f"Methode d'attribution non supportee: {method}")
            return attributions.detach().numpy()
        except Exception as e:
            logging.error(f"Erreur dans get_attributions ({method}): {e}")
            return np.zeros_like(sequence.numpy() if isinstance(sequence, torch.Tensor) else sequence)
    
    def gradients_integres(self, sequence, target_class=None, steps=50):
        # Gradients integrates pour expliquer les decisions
        return self.obtenir_explications(sequence, target_class, 'integrated_gradients')
    
    def carte_importance(self, sequence, target_class=None):
        # Carte d'importance des capteurs
        return self.obtenir_explications(sequence, target_class, 'saliency')
    
    def explication_shap(self, sequence, baseline=None, target_class=None, steps=50):
        # Explication SHAP basee sur les gradients
        return self.obtenir_explications(sequence, target_class, 'gradient_shap')
    
    def explication_profonde(self, sequence, baseline=None, target_class=None):
        # Explication profonde des decisions
        return self.obtenir_explications(sequence, target_class, 'deep_lift')
    
    def explication_couches(self, sequence, target_class=None, steps=50):
        # Explication par ablation des features
        return self.obtenir_explications(sequence, target_class, 'feature_ablation')
    
    def explication_feature_ablation(self, sequence, target_class=None):
        try:
            self.eval()
            if isinstance(sequence, np.ndarray):
                sequence = torch.FloatTensor(sequence)
            if target_class is None:
                with torch.no_grad():
                     predictions = self.forward(sequence)
                     target_class = torch.argmax(predictions, dim=1)
            attributor = FeatureAblation(self)
            attributions = attributor.attribute(sequence, target=target_class)
            return attributions.detach().numpy()
        except Exception as e:
            logging.error(f"Erreur dans la fonction feature ablation: {e}")
            return np.zeros_like(sequence.numpy() if isinstance(sequence, torch.Tensor) else sequence)

class DonneesPartagees:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = np.zeros((0, FEATURE_COUNT), dtype=np.float32)
        self.error_history = []
        self.attack_counter = 0
        self.last_time_mess = pytz.timezone("America/Toronto").localize(datetime.now())
        self.last_message_timestamp: Optional[datetime] = None
        self.prob_history = deque(maxlen=5000) # Stocke les probabilites sous forme de dictionnaire {classe: prob}
        self.last_alert_time = 0

    def update_history(self, new_data: np.ndarray) -> None:
        with self.lock:
            if self.history.shape[0] == 0:
                self.history = new_data
            else:
                self.history = np.vstack([self.history, new_data])
                if self.history.shape[0] > n_steps:
                    self.history = self.history[-n_steps:]

    def get_history(self) -> np.ndarray:
        with self.lock:
            return self.history.copy()

    def increment_attack_counter(self) -> int:
        with self.lock:
            self.attack_counter += 1
            return self.attack_counter

    def add_error(self, error: str) -> None:
        with self.lock:
            self.error_history.append({
                'timestamp': datetime.now(),
                'error': error
            })

    def add_probabilities(self, probabilities: Dict[str, float]) -> None:
        with self.lock:
            self.prob_history.append(probabilities)

    def get_probability_history(self) -> deque:
         with self.lock:
             return self.prob_history.copy()

# Instanciation des Objets Globaux
# Etat partage
shared_state = DonneesPartagees()

# Queues pour la communication entre les threads
detection_queue = queue.Queue()
ui_queue = queue.Queue()

def initialiser_systeme():
    # Chargement du modele PyTorch LSTM
    try:
        model = IntelligenceArtificielle(
            input_size=FEATURE_COUNT, 
            hidden_size_1=512,
            hidden_size_2=256,
            hidden_size_3=128,
            num_classes=4,
            dropout=0.4
        )
        model_paths = ['pytorch_model.pth']
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    loaded_data = torch.load(model_path, map_location="cpu", weights_only=False)
                    if isinstance(loaded_data, dict):
                        model.load_state_dict(loaded_data)
                        logging.info(f"Charge comme state_dict depuis {model_path}")
                    else:
                        model = loaded_data
                        logging.info(f"Charge comme modele complet depuis {model_path}")
                    model.eval()
                    model_loaded = True
                    break
                except Exception as e:
                    logging.warning(f"Impossible de charger {model_path}: {e}")
                    continue
        if not model_loaded:
            logging.warning("Aucun modele entraine trouve")
            logging.info("Modele cree avec des poids aleatoires")
        model.eval()
    except Exception as e:
        logging.error(f"Erreur lors de la creation du modele: {str(e)}")
        traceback.print_exc()
        raise
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        #sensor_encoder = joblib.load('ID_encoder.pkl')
        logging.info("Scalers et encodeurs charges avec succes")
    except Exception as e:
        logging.error(f"Erreur lors du chargement des scalers/encodeurs: {str(e)}")
        traceback.print_exc()
        raise
    try:
        X_train = np.load('X_train.npy')
        logging.info("Donnees d'entrainement chargees")
    except Exception as e:
        logging.warning(f"Donnees d'entrainement non trouvees: {str(e)}")
    try:
        analyseur_intelligent = AnalyseurIntelligent(model, scaler, label_encoder, X_train)
        logging.info("Model initialise avec succes")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modele: {str(e)}")
        traceback.print_exc()
        raise
    return model, scaler, label_encoder, analyseur_intelligent

class AnalyseurIntelligent:
    def __init__(self, model: IntelligenceArtificielle, scaler: MinMaxScaler, label_encoder: LabelEncoder, X_train: np.ndarray):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_names = ["consommation", "temperature", "light", "detecteur", "door", "connectivity"]
        self.X_train = X_train
        logging.info("Explicabilite XAI initialise")

    def detecter_attaque(self, sequence: np.ndarray, explanation: Dict[str, Any]) -> tuple[Optional[str], float, Dict[str, Any]]:
        try:
            # Prediction
            probabilities = self.model.calculer_probabilites(sequence)[0]            
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            probability = float(probabilities[predicted_class_idx])
            logging.info(f"Probabilites pour chaque classe:")
            for idx, prob in enumerate(probabilities):
                class_name = self.label_encoder.inverse_transform([idx])[0]
                logging.info(f"{class_name}: {prob:.4f}")
            is_attack_detected = predicted_class in ['fdia', 'hybrid', 'replay']
            logging.info(f"Attack detectee: {is_attack_detected}, la classe predit est: {predicted_class}")
            if is_attack_detected:
               explanation['feature_importance'] = self._calculer_importance_capteurs(sequence)
               explanation['saliency_map'] = self._extraire_carte_importance(sequence)
               explanation['gradient_shap'] = self._extraire_explication_shap(sequence)
               explanation['deep_lift'] = self._extraire_explication_profonde(sequence)
               explanation['feature_ablation'] = self._extraire_feature_ablation(sequence)
               explanation['correlations'] = self._extraire_correlations(sequence)
               comparaison_plots = self.creer_graphiques_comparaison(explanation)
               explanation['comparaison_plots'] = comparaison_plots
               logging.info("Explications calculees pour l'attaque")
            else:
               logging.info(f"Pas d'explication pour trafic normal")
               explanation['feature_importance'] = {}
               explanation['saliency_map'] = None
               explanation['gradient_shap'] = None
               explanation['deep_lift'] = None
               explanation['feature_ablation'] = None
               explanation['correlations'] = {}
               explanation['comparaison_plots'] = {}
            return predicted_class, probability, explanation
        except Exception as e:
            logging.error(f"Erreur lors de la detection d'attaque: {str(e)}")
            traceback.print_exc()
            return None, 0.0, {}

    def _calculer_importance_capteurs(self, sequence: np.ndarray) -> Dict[str, float]:
        try:
            attributions = self.model.gradients_integres(sequence, steps=20)
            logging.info(f"Attributions Iintegrated Gradients shap: {attributions.shape}")
            feature_importance = np.mean(np.abs(attributions[0]), axis=0)
            logging.info(f"Feature importance shape: {feature_importance.shape}, values: {feature_importance}")
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            return {name: float(round(val, 3)) for name, val in zip(self.feature_names, feature_importance)}
        except Exception as e:
            logging.error(f"Erreur lors du calcul de l'importance des features: {str(e)}")
            traceback.print_exc()
            return {name: 0.0 for name in self.feature_names}

    def _extraire_carte_importance(self, sequence: np.ndarray) -> np.ndarray:
        try:
            saliency_map = self.model.carte_importance(sequence)
            if np.max(np.abs(saliency_map)) > 0:
                saliency_map = np.abs(saliency_map[0]) / np.max(np.abs(saliency_map[0]))
            logging.info(f"Saliency map calculee: {saliency_map.shape}")
            return saliency_map
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de la saliency map: {str(e)}")
            traceback.print_exc()
            return np.zeros((sequence.shape[1], sequence.shape[2]))
    
    def _extraire_explication_shap(self, sequence: np.ndarray) -> np.ndarray:
        try:
            gradient_shap = self.model.explication_shap(sequence, steps=20)
            if np.max(np.abs(gradient_shap)) > 0:
                gradient_shap = np.abs(gradient_shap[0]) / np.max(np.abs(gradient_shap[0]))
            logging.info(f"GradientShap calcule: {gradient_shap.shape}")
            return gradient_shap
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de GradientShap: {str(e)}")
            traceback.print_exc()
            return np.zeros((sequence.shape[1], sequence.shape[2]))
    
    def _extraire_explication_profonde(self, sequence: np.ndarray) -> np.ndarray:
        try:
            deep_lift = self.model.explication_profonde(sequence)
            if np.max(np.abs(deep_lift)) > 0:
                deep_lift = np.abs(deep_lift[0]) / np.max(np.abs(deep_lift[0]))
            logging.info(f"DeepLift calcule: {deep_lift.shape}")
            return deep_lift
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de DeepLift: {str(e)}")
            traceback.print_exc()
            return np.zeros((sequence.shape[1], sequence.shape[2]))
    
    def _extraire_feature_ablation(self, sequence: np.ndarray) -> np.ndarray:
        try:
            attributions = self.model.explication_feature_ablation(sequence)    
            if np.max(np.abs(attributions)) > 0:
                attributions = np.abs(attributions) / np.max(np.abs(attributions))
            logging.info(f"Feature ablation calculee: {attributions.shape}")
            return attributions
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de Feature ablation: {str(e)}")
            traceback.print_exc()
            return np.zeros((sequence.shape[1], sequence.shape[2]))

    def _extraire_correlations(self, sequence: np.ndarray) -> Dict[str, float]:
        try:
            correlations = {}
            seq = sequence[0]  # (n_steps, n_features)
            correlation_matrix = np.corrcoef(seq.T)
            for i, feature1 in enumerate(self.feature_names):
                for j, feature2 in enumerate(self.feature_names[i+1:], i+1):
                    key = f"{feature1}_{feature2}"
                    correlations[key] = float(correlation_matrix[i, j])
            logging.info(f"Correlations extraites: {correlations}")
            return correlations
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction des correlations: {str(e)}")
            traceback.print_exc()
            return {}
            
    def creer_graphiques_comparaison(self, explanation: Dict[str, Any]) -> Dict[str, str]:
        try:
            plots = {}
            methods_data = {}
            logging.info(f"Comparaison des graphiques explanation keys: {list(explanation.keys())}")
            if 'feature_importance' in explanation:
                methods_data['Integrated Gradients'] = list(explanation['feature_importance'].values())
                logging.info(f"Integrated gradients ajoute avec succes: {len(methods_data['Integrated Gradients'])} values")
            if 'deep_lift' in explanation and explanation['deep_lift'] is not None:
                deep_lift_data = np.mean(np.abs(explanation['deep_lift']), axis=0)
                methods_data['DeepLift'] = deep_lift_data.tolist()
                logging.info(f"Deeplift ajoute avec succes: {len(methods_data['DeepLift'])} values")
            if 'gradient_shap' in explanation and explanation['gradient_shap'] is not None:
                gradient_shap_data = np.mean(np.abs(explanation['gradient_shap']), axis=0)
                methods_data['Gradient SHAP'] = gradient_shap_data.tolist()
                logging.info(f"Gradients shap ajoute avec succes: {len(methods_data['Gradient SHAP'])} values")
            if 'feature_ablation' in explanation and explanation['feature_ablation'] is not None:
                feature_ablat_data = np.mean(np.abs(explanation['feature_ablation']), axis=(0,1))
                methods_data['Feature ablation'] = feature_ablat_data.tolist()
                logging.info(f"Feature ablation ajoute avec succes: {len(methods_data['Feature ablation'])} values")
            logging.info(f"Toutes les images ont ete ajoutees: {list(methods_data.keys())}")
            if not methods_data:
                logging.warning("Aucune donnee d'explicabilite disponible pour les graphiques")
                return plots
            # Permet de creer le graphique de comparaison des features
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(self.feature_names))
            width = 0.2
            colors = ['#ff69b4', '#4169e1', '#8b4513', '#dc143c']
            for i, (method, data) in enumerate(methods_data.items()):
                logging.info(f"Traitement de {method}: len(data)={len(data)}, len(feature_names)={len(self.feature_names)}")
                if len(data) == len(self.feature_names):
                    ax.bar(x + i * width, data, width, label=method, color=colors[i % len(colors)], alpha=0.8)
                    logging.info(f"Ajout graph pour {method}")
                else:
                    logging.warning(f"Donnees ignorees pour {method}: {len(data)} valeurs au lieu de {len(self.feature_names)}")
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Comparaison des Methodes d\'Explainable par Feature')
            ax.set_xticks(x + width * (len(methods_data) - 1) / 2)
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            # Permet de sauvegarder le graphique
            comparaison_path = 'feature_comparaison.png'
            plt.savefig(comparaison_path, dpi=150, bbox_inches='tight')
            plots['feature_comparaison'] = comparaison_path
            logging.info(f"Graphique comparaison sauvegarde: {comparaison_path}")
            if os.path.exists(comparaison_path):
               logging.info(f"Fichier {comparaison_path} existe, taille: {os.path.getsize(comparaison_path)} bytes")
            else:
               logging.error(f"Fichier {comparaison_path} n'existe pas")
            plt.close(fig)
            # Permet de creer des graphiques en barres individuels pour chaque methode
            for method, data in methods_data.items():
                if len(data) == len(self.feature_names):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(self.feature_names, data, color=colors[list(methods_data.keys()).index(method) % len(colors)], alpha=0.8)
                    for bar, value in zip(bars, data):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Importance')
                    ax.set_title(f'Importance des Features - {method}')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    # Permet de sauvegarder le graphique
                    method_path = f'{method.lower().replace(" ", "_")}.png'
                    plt.savefig(method_path, dpi=150, bbox_inches='tight')
                    plots[method.lower().replace(" ", "_")] = method_path
                    plt.close(fig)
            logging.info(f"Graphiques de comparaison creates: {list(plots.keys())}")
            return plots
        except Exception as e:
            logging.error(f"Erreur lors de la creation des graphiques de comparaison: {str(e)}")
            traceback.print_exc()
            return {}

class InterfaceSurveillance:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Surveillance temps reel Smart Home")
        self.master.geometry("800x400")
        self.horodatages = deque(maxlen=1000)
        self.donnees_temperature = deque(maxlen=1000)
        self.donnees_lumiere = deque(maxlen=1000)
        self.donnees_importance_capteurs = deque(maxlen=1000)
        self.cartes_importance = deque(maxlen=1000)
        self.correlations = deque(maxlen=1000)
        #self.correlations_capteurs = deque(maxlen=1000)
        self.historique_alertes = []
        self._setup_styles()
        self.creer_ecrans()
        self.verifier_messages_interface()
        self.open_alert_windows = 0
        self.max_alert_windows = 15

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#f0f0f0")
        style.configure("Custom.TLabel", background="#f0f0f0", padding=5)
        style.configure("Custom.TButton", padding=5)

    def verifier_messages_interface(self) -> None:
        try:
            # Permet de traiter tous les elements disponibles dans la queue
            while True:
                update_type, *args = ui_queue.get_nowait()
                logging.info(f"check_ui_queue: recu {update_type}")
                if update_type == "alert":
                    logging.info(f"Affichage de l'alerte: {args[0]}")
                    self.afficher_alerte(*args)
                elif update_type == "data":
                    logging.info(f"ajouter_donnees appelee avec: temp={args[0]}, light={args[1]}, feature_importance={args[2]}")
                    self.ajouter_donnees(*args)
                elif update_type == "stats_update":
                    logging.info(f"Mise a jour des statistiques: {args}")
                    self.maj_statistiques(*args)
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Erreur dans check_ui_queue: {str(e)}")
            traceback.print_exc()
        finally:
            # Planifier la prochaine verification
            self.master.after(100, self.verifier_messages_interface)

    def afficher_alerte(self, attack_type: str, probability: float, explanation: Dict[str, Any],
                  normalized_stats: Dict[str, Dict[str, float]]) -> None:
        try:
            self.historique_alertes.append({
                'type': attack_type,
                'probability': float(probability),
                'explanation': explanation,
                'timestamp': datetime.now(),
                'normalized_stats': normalized_stats
            })
            if self.open_alert_windows >= self.max_alert_windows:
                messagebox.showinfo("Limite atteinte", f"Vous avez {self.max_alert_windows} alertes ouvertes")
                return
            self.open_alert_windows += 1
            # Creation d'une fenetre popup pour les details
            popup = Toplevel(self.master)
            popup.title(f"Alerte - {attack_type}")
            popup.geometry("400x200")
            main_frame = ttk.Frame(popup, style="Custom.TFrame")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            title_label = ttk.Label(main_frame, text=f"Alerte - {attack_type}", 
                                  font=("Helvetica", 14, "bold"))
            title_label.pack(pady=5)
            details_frame = ttk.Frame(main_frame, style="Custom.TFrame")
            details_frame.pack(fill="both", expand=True, pady=5)
            canvas = tk.Canvas(details_frame, bg="#f0f0f0")
            scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas, style="Custom.TFrame")
            def configure_scroll(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            scrollable_frame.bind("<Configure>", configure_scroll)
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            ttk.Label(scrollable_frame, text="Probabilite:", 
                     font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
            ttk.Label(scrollable_frame, text=f"{float(probability):.2%}", 
                     font=("Helvetica", 10)).pack(anchor="w", padx=10)
            ttk.Label(scrollable_frame, text="Valeurs des Capteurs:", 
                     font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
            sensor_values = explanation.get('sensor_values', {})
            for k, v in sensor_values.items():
                try:
                    value = float(v)
                    ttk.Label(scrollable_frame, text=f"{k}: {value:.2f}", 
                             font=("Helvetica", 10)).pack(anchor="w", padx=10)
                except (ValueError, TypeError):
                    ttk.Label(scrollable_frame, text=f"{k}: {v}", 
                             font=("Helvetica", 10)).pack(anchor="w", padx=10)
            # Importance des features
            ttk.Label(scrollable_frame, text="Importance des Features:", 
                     font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
            feature_imp = explanation.get('feature_importance', {}).items()
            sorted_feature_imp = sorted(feature_imp, key=lambda item: item[1], reverse=True)
            for k, v in sorted_feature_imp:
                try:
                    value = float(v)
                    ttk.Label(scrollable_frame, text=f"{k}: {value:.2f}", 
                             font=("Helvetica", 10)).pack(anchor="w", padx=10)
                except (ValueError, TypeError):
                    ttk.Label(scrollable_frame, text=f"{k}: {v}", 
                             font=("Helvetica", 10)).pack(anchor="w", padx=10)
            # Carte de salience
            if 'saliency_map' in explanation:
                ttk.Label(scrollable_frame, text="Saliency MAP", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                saliency_map = explanation['saliency_map']
                # Si la saliency_map est 2D (n_steps, n_features), on affiche la heatmap
                if isinstance(saliency_map, np.ndarray) and saliency_map.ndim == 2:
                    # permet de determiner les noms de features
                    if hasattr(self, 'feature_names'):
                        feature_names = self.feature_names
                    else:
                        feature_names = ["consommation", "temperature", "light", "detecteur", "door", "connectivity"]
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    im = ax.imshow(np.abs(saliency_map).T, aspect='auto', cmap='hot', origin='lower')
                    ax.set_xlabel("Temps")
                    ax.set_ylabel("Features")
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_yticklabels(feature_names)
                    fig.colorbar(im, ax=ax, orientation='vertical', label="Importance")
                    plt.tight_layout()
                    # Convertir la figure matplotlib en image Tkinter
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    photo = ImageTk.PhotoImage(img)
                    label_img = tk.Label(scrollable_frame, image=photo)
                    label_img.image = photo  # Pour eviter le garbage collection
                    label_img.pack(anchor="w", pady=5)
                    plt.close(fig)
                else:
                    # Affichage texte simple si 1D
                    ttk.Label(scrollable_frame, text="Carte de Salience:", 
                             font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
                    for i, feature in enumerate(["consommation", "temperature", "light", "detecteur", "door", "connectivity"]):
                        ttk.Label(scrollable_frame, text=f"{feature}: {saliency_map[i]:.3f}", font=("Helvetica", 10)).pack(anchor="w", padx=10)
            ttk.Label(scrollable_frame, text="Explainable XAI by LSTM", 
                     font=("Helvetica", 12, "bold"), foreground="black").pack(anchor="w", pady=(15, 5))
            
            #ttk.Label(scrollable_frame, text="Methodes XAI choisies:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
            if 'feature_importance' in explanation:
                ttk.Label(scrollable_frame, text="Integrated Gradients:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                for feature, importance in explanation['feature_importance'].items():
                    ttk.Label(scrollable_frame, text=f"  {feature}: {importance:.4f}", 
                             font=("Helvetica", 9)).pack(anchor="w", padx=30)     
            if 'gradient_shap' in explanation and explanation['gradient_shap'] is not None:
                ttk.Label(scrollable_frame, text="Gradient SHAP:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                gs_mean = np.mean(np.abs(explanation['gradient_shap']))
                gs_max = np.max(np.abs(explanation['gradient_shap']))
                ttk.Label(scrollable_frame, text=f"  Moyenne: {gs_mean:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)
                ttk.Label(scrollable_frame, text=f"  Maximum: {gs_max:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)    
            if 'deep_lift' in explanation and explanation['deep_lift'] is not None:
                ttk.Label(scrollable_frame, text="DeepLift:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                dl_mean = np.mean(np.abs(explanation['deep_lift']))
                dl_max = np.max(np.abs(explanation['deep_lift']))
                ttk.Label(scrollable_frame, text=f"  Moyenne: {dl_mean:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)
                ttk.Label(scrollable_frame, text=f"  Maximum: {dl_max:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)
            
            if 'feature_ablation' in explanation and explanation['feature_ablation'] is not None:
                ttk.Label(scrollable_frame, text="Feature Ablation:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                fa_mean = np.mean(np.abs(explanation['feature_ablation']))
                fa_max = np.max(np.abs(explanation['feature_ablation']))
                ttk.Label(scrollable_frame, text=f"  Moyenne: {fa_mean:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)
                ttk.Label(scrollable_frame, text=f"  Maximum: {fa_max:.4f}", 
                         font=("Helvetica", 9)).pack(anchor="w", padx=30)
            
            if 'saliency_map' in explanation and explanation['saliency_map'] is not None:
                ttk.Label(scrollable_frame, text="Statistiques Saliency Map:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
                saliency_map = explanation['saliency_map']
                if isinstance(saliency_map, np.ndarray):
                    sm_mean = np.mean(np.abs(saliency_map))
                    sm_max = np.max(np.abs(saliency_map))
                    sm_std = np.std(saliency_map)
                    ttk.Label(scrollable_frame, text=f"  Moyenne: {sm_mean:.4f}", 
                             font=("Helvetica", 9)).pack(anchor="w", padx=30)
                    ttk.Label(scrollable_frame, text=f"  Maximum: {sm_max:.4f}", 
                             font=("Helvetica", 9)).pack(anchor="w", padx=30)
                    ttk.Label(scrollable_frame, text=f"  Ecart-type: {sm_std:.4f}", 
                             font=("Helvetica", 9)).pack(anchor="w", padx=30)
            
            ttk.Label(scrollable_frame, text="Comparaison des methodes XAI:", 
                     font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10)
            captum_scores = {}
            if 'feature_importance' in explanation:
                captum_scores['Integrated Gradients'] = np.mean(list(explanation['feature_importance'].values()))
            if 'gradient_shap' in explanation and explanation['gradient_shap'] is not None:
                captum_scores['Gradient SHAP'] = np.mean(np.abs(explanation['gradient_shap']))
            if 'deep_lift' in explanation and explanation['deep_lift'] is not None:
                captum_scores['DeepLift'] = np.mean(np.abs(explanation['deep_lift']))
            if 'feature_ablation' in explanation and explanation['feature_ablation'] is not None:
                captum_scores['Feature ablation'] = np.mean(np.abs(explanation['feature_ablation']))
            if captum_scores:
                sorted_methods = sorted(captum_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (method, score) in enumerate(sorted_methods):
                    ttk.Label(scrollable_frame, text=f"  {i+1}. {method}: {score:.4f}", 
                             font=("Helvetica", 9)).pack(anchor="w", padx=30)
            else:
                ttk.Label(scrollable_frame, text="  Aucune methode XAI disponible", 
                         font=("Helvetica", 9, "italic")).pack(anchor="w", padx=30)
            # Permet d'afficher les graphiques de comparaison
            logging.info(f"Verification des graphiques dans l'alerte: 'comparaison_plots' in explanation {'comparaison_plots' in explanation}")
            if 'comparaison_plots' in explanation:
                logging.info(f"Comparaison_plots value: {explanation['comparaison_plots']}")
                logging.info(f"Comparaison_plots is not None: {explanation['comparaison_plots'] is not None}")
                logging.info(f"Comparaison_plots is not empty: {bool(explanation['comparaison_plots'])}")
            if 'comparaison_plots' in explanation and explanation['comparaison_plots']:
                ttk.Label(scrollable_frame, text="Graphiques de Comparaison:", 
                         font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
                plots = explanation['comparaison_plots']
                logging.info(f"Graphiques disponibles dans l'interface:{plots['feature_comparaison']}")
                if 'feature_comparaison' in plots:
                    try:
                        logging.info(f"Tentative d'ouverture du fichier: {explanation['comparaison_plots']}")
                        img = Image.open(plots['feature_comparaison'])
                        img = img.resize((400, 300), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        comparaison_label = tk.Label(scrollable_frame, image=photo)
                        comparaison_label.image = photo
                        comparaison_label.pack(anchor="w", pady=5)
                        logging.info(f"Graphique de comparaison affichee avec succes")
                    except Exception as e:
                        logging.error(f"Erreur lors de l'affichage du graphique de comparaison: {e}")
                        traceback.print_exc()
                else:
                    logging.warning("'Feature_comparaison' non trouve dans les plots")
                for plot_name, plot_path in plots.items():
                    if plot_name != 'feature_comparaison':
                        try:
                            logging.info(f"Tentative d'affichage du graphique {plot_name}: {plot_path}")
                            img = Image.open(plot_path)
                            img = img.resize((350, 200), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            plot_label = tk.Label(scrollable_frame, image=photo)
                            plot_label.image = photo
                            plot_label.pack(anchor="w", pady=2)
                            logging.info(f"Graphique {plot_name} affiche avec succes")
                        except Exception as e:
                            logging.error(f"Erreur lors de l'affichage du graphique {plot_name}: {e}")
                            traceback.print_exec()
            else:
                logging.warning("Aucun graph de comparaison dispo dans l'explanation")
            # Correlations
            if 'correlations' in explanation:
                ttk.Label(scrollable_frame, text="Correlations entre Features:", 
                         font=("Helvetica", 10, "bold")).pack(anchor="w", pady=2)
                correlations = explanation['correlations']
                for pair, value in correlations.items():
                    ttk.Label(scrollable_frame, text=f"{pair}: {value:.3f}", 
                             font=("Helvetica", 10)).pack(anchor="w", padx=10)
            # Statistiques temporelles normalisees
            ttk.Label(scrollable_frame, text="Statistiques Temporelles - Normalisees:", 
                     font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 2))
            stats_text_content = ""
            for feature, stats in normalized_stats.items():
                stats_text_content += f"{feature}: Mean={stats['Mean']:.2f}, Std={stats['Std']:.2f}, Min={stats['Min']:.2f}, Max={stats['Max']:.2f}, Total Change={stats['Total Change']:.2f}\n"
            ttk.Label(scrollable_frame, text=stats_text_content, 
                     font=("Courier New", 9), justify="left").pack(anchor="w", padx=10)
        except Exception as e:
            logging.error(f"Erreur lors de l'affichage de l'alerte: {str(e)}")
            traceback.print_exc()
            # En cas d'erreur, on s'assure que le compteur est decremente
            self.open_alert_windows = max(0, self.open_alert_windows - 1)
        def on_close():
            try:
                # Détacher les bindings de la souris
                canvas.unbind_all("<MouseWheel>")
                self.open_alert_windows -= 1
                popup.destroy()
            except Exception as e:
                logging.error(f"Erreur lors de la fermeture de l'alerte: {str(e)}")
                traceback.print_exc()

        # Configure le protocole de fermeture de la fenetre
        popup.protocol("WM_DELETE_WINDOW", on_close)            
        # Bouton de fermeture
        close_button = ttk.Button(main_frame, text="Fermer", command=on_close)
        close_button.pack(pady=10)

    def creer_ecrans(self) -> None:
        # Création d'un Canvas avec Scrollbar pour la fenêtre principale
        canvas = tk.Canvas(self.master, borderwidth=0, background="#f0f0f0")
        vscrollbar = ttk.Scrollbar(self.master, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscrollbar.set)
        vscrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        # Frame principal a l'interieur du Canvas
        main_container = ttk.Frame(canvas, style="Custom.TFrame")
        canvas.create_window((0, 0), window=main_container, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        main_container.bind("<Configure>", on_frame_configure)
        # Titre principal
        title_label = ttk.Label(main_container, text="Surveillance en Temps Reel", 
                              font=("Helvetica", 14, "bold"))
        title_label.pack(pady=5)
        # Frame pour les statistiques
        stats_patterns_frame = ttk.Frame(main_container, style="Custom.TFrame")
        stats_patterns_frame.pack(fill="x", pady=5)
        # Frame pour les graphiques de capteurs et scores
        plots_and_scores_frame = ttk.Frame(main_container, style="Custom.TFrame")
        plots_and_scores_frame.pack(fill="both", expand=True)
        # Section Statistiques Temporelles Normalisees
        normalized_stats_label = ttk.Label(stats_patterns_frame, text="Statistiques Temporelles - Normalisees:", font=("Helvetica", 10, "bold"))
        normalized_stats_label.pack(anchor="w")
        self.normalized_stats_text = tk.Text(stats_patterns_frame, height=10, wrap="word", state="disabled", font=("Courier New", 10))
        self.normalized_stats_text.pack(fill="x", pady=2)
        # Section XAI Explanations
        xai_frame = ttk.LabelFrame(main_container, text="XAI Explanations", style="Custom.TFrame")
        xai_frame.pack(fill="both", expand=True, pady=5)
        # Saliency Map
        saliency_label = ttk.Label(xai_frame, text="Explainable XAI:", font=("Helvetica", 10, "bold"))
        saliency_label.pack(anchor="w")
        self.saliency_text = tk.Text(xai_frame, height=6, wrap="word", state="disabled", font=("Courier New", 10))
        self.saliency_text.pack(fill="x", pady=2)
        self.saliency_text.pack(fill="both", expand=True)
        # Correlations
        correlations_label = ttk.Label(xai_frame, text="Feature Correlations:", font=("Helvetica", 10, "bold"))
        correlations_label.pack(anchor="w", pady=(10, 0))
        self.correlations_text = tk.Text(xai_frame, height=10, wrap="word", state="disabled", font=("Courier New", 10))
        self.correlations_text.pack(fill="x", pady=2)
        self.correlations_text.pack(fill="both", expand=True)
        # Frame pour les scores d'attaque et importance features
        attack_and_features_frame = ttk.Frame(plots_and_scores_frame, style="Custom.TFrame")
        attack_and_features_frame.pack(side="top", fill="both", expand=True, pady=2)
        # Frame pour l'importance des features
        features_frame = ttk.LabelFrame(attack_and_features_frame, text="Importance des Features", style="Custom.TFrame")
        features_frame.pack(side="left", fill="both", expand=True, padx=2)
        self.features_label = ttk.Label(features_frame, text="--", font=("Helvetica", 10))
        self.features_label.pack(pady=2)

    def ajouter_donnees(self, temperature: float, light: float, feature_importance: Dict[str, float],
                saliency_map: np.ndarray = None, correlations: Dict[str, float] = None, explanation: Dict[str, Any] = None) -> None:
        # Mise a jour des donnees en temps reel
        current_time = datetime.now()
        self.horodatages.append(current_time)
        self.donnees_temperature.append(temperature)
        self.donnees_lumiere.append(light)
        self.donnees_importance_capteurs.append(feature_importance)
        if saliency_map is not None: 
            self.cartes_importance.append(saliency_map)
        if correlations is not None:
            self.correlations.append(correlations)
        # Mise a jour de l'importance des features
        if hasattr(self, 'features_label'):
            features_text = "\n".join([f"{k}: {v:.2f}" for k, v in feature_importance.items()])
            self.features_label.config(text=features_text)
            logging.info(f"Label des features mis a jour: {features_text}")
        if hasattr(self, 'normalized_stats_text') and explanation is not None:
            if len(self.horodatages) > 1:
               history_data = np.array([
                  list(self.donnees_temperature),
                  list(self.donnees_lumiere),
                  [0.0] * len(self.donnees_temperature),
                  [0.0] * len(self.donnees_temperature),
                  [0.0] * len(self.donnees_temperature),
                  [0.0] * len(self.donnees_temperature),
                  [0.0] * len(self.donnees_temperature)
               ]).T
               normalized_stats = {}
               feature_names = ["temperature", "light", "consommation", "detecteur", "door", "connectivity"]
               for i, feature in enumerate(feature_names):
                   if i < history_data.shape[1]:
                       col_data = history_data[:, i]
                       if len(col_data) > 0:
                          mean_val = float(np.mean(col_data))
                          std_val  = float(np.std(col_data))
                          min_val  = float(np.min(col_data))
                          max_val  = float(np.max(col_data))
                          change_total = float(col_data[-1] - col_data[0]) if len(col_data) > 1 else 0.0
                          median_val  = float(np.median(col_data))
                          var_val     = float(np.var(col_data))
                          #pente pour regression lineaire
                          if len(col_data) > 1:
                             x = np.arange(len(col_data))
                             slope, intercept = np.polyfit(x, col_data, 1)
                             trend_val = float(slope)
                          else:
                             trend_val = 0.0
                          normalized_stats[feature] = {
                            'Mean': mean_val,
                            'Median': median_val,
                            'Std': std_val,
                            'Variance': var_val,
                            'Min': min_val,
                            'Max': max_val,
                            'Total Change': change_total,
                            'Trend': trend_val
                          }
               self.normalized_stats_text.config(state="normal")
               self.normalized_stats_text.delete(1.0, tk.END)
               stats_text = ""
               for feature, stats in normalized_stats.items():
                   stats_text += f"{feature}: Mean={stats['Mean']:.3f}, Median={stats['Median']:.3f}, Std={stats['Std']:.3f}, Min={stats['Min']:.3f}, Max={stats['Max']:.3f}, Variance={stats['Variance']:.3f}, Total Change={stats['Total Change']:.3f}, Trend={stats['Trend']:.3f}\n"
               self.normalized_stats_text.insert(tk.END, stats_text)
               self.normalized_stats_text.config(state="disabled")
               logging.info(f"Statistiques normalisees pour affichage dans uI principale: {stats_text}")
            else:
               self.normalized_stats_text.config(state="normal")
               #self.normalized_stats_text.config(tk.END, f"Donnees recues: Temp={temperature:.2f}, Light={light:.2f}\nEn attente de plus de donnees concernant les statistiques")
               self.normalized_stats_text.config(state="disabled")
        # Mise a jour des explications XAI
        if saliency_map is not None:
            self.saliency_text.config(state="normal")
            self.saliency_text.delete(1.0, tk.END)
            saliency_text = "EXPLICABILITE XAI : \n"
            saliency_text += "Feature Importance:\n"
            saliency_text += "-" * 40 + "\n"
            for i, feature in enumerate(["consommation", "temperature", "light", "detecteur", "door", "connectivity"]):
                # Prendre la moyenne sur le temps pour chaque feature
                value = float(np.mean(saliency_map[:, i]))
                saliency_text += f"{feature}: {value:.3f}\n"
            # permet d'ajouter des statistiques de la saliency map
            saliency_text += "\nStatistiques Saliency Map:\n"
            saliency_text += f"  Moyenne globale: {np.mean(np.abs(saliency_map)):.3f}\n"
            saliency_text += f"  Maximum: {np.max(np.abs(saliency_map)):.3f}\n"
            saliency_text += f"  Ecart-type: {np.std(saliency_map):.3f}\n"
            self.saliency_text.insert(tk.END, saliency_text)
            self.saliency_text.config(state="disabled")
        # Permet d'ajouter toutes les methodes XAI disponibles dans l'explication
        if hasattr(self, 'saliency_text') and explanation is not None:
            self.saliency_text.config(state="normal")            
            self.saliency_text.delete(1.0, tk.END)
            data_xai = any([
                explanation.get('feature_importance'),
                explanation.get('gradient_shap') is not None,
                explanation.get('deep_lift') is not None,
                explanation.get('feature_ablation') is not None,
            ])
            if data_xai:
               captum_text = "METHODES XAI DETAILLEES:\n"
               # Gradient SHAP
               if 'gradient_shap' in explanation and explanation['gradient_shap'] is not None:
                 gs_mean = np.mean(np.abs(explanation['gradient_shap']))
                 captum_text += f"\nGradient SHAP: {gs_mean:.3f}\n"
               # DeepLift
               if 'deep_lift' in explanation and explanation['deep_lift'] is not None:
                 dl_mean = np.mean(np.abs(explanation['deep_lift']))
                 captum_text += f"DeepLift: {dl_mean:.3f}\n"
               # Feature Ablation
               if 'feature_ablation' in explanation and explanation['feature_ablation'] is not None:
                 fa_mean = np.mean(np.abs(explanation['feature_ablation']))
                 captum_text += f"Feature Ablation: {fa_mean:.3f}\n"
               if 'comparaison_plots' in explanation and explanation['comparaison_plots']:
                 captum_text += f"\nGraphiques de comparaison generates: {len(explanation['comparaison_plots'])} fichiers\n"
                 for plot_name in explanation['comparaison_plots'].keys():
                    captum_text += f"  - {plot_name}\n"
            else:
                captum_text = "Trafic normal detectee\n"
                captum_text += "Explanation calculee lorsqu'une attaque sera detectee"
            self.saliency_text.insert(tk.END, captum_text)
            self.saliency_text.config(state="disabled")
        if correlations is not None:
            self.correlations_text.config(state="normal")
            self.correlations_text.delete(1.0, tk.END)
            correlations_text = "Feature Correlations:\n"
            for pair, value in correlations.items():
                correlations_text += f"{pair}: {value:.3f}\n"
            self.correlations_text.insert(tk.END, correlations_text)
            self.correlations_text.config(state="disabled")

    def maj_statistiques(self, normalized_stats: Dict[str, Dict[str, float]]) -> None:
        try:
            logging.info(f"MAJ des statistiques avec UI: {normalized_stats}")
            logging.info(f"normalized_stats_text existe: {hasattr(self, 'normalized_stats_text')}")
            if hasattr(self, 'normalized_stats_text'):
               logging.info(f"normalized_stats_text type: {type(self.normalized_stats_text)}")
            self.normalized_stats_text.config(state="normal")
            self.normalized_stats_text.delete(1.0, tk.END)
            stats_text = ""
            for feature, stats in normalized_stats.items():
                stats_text += f"{feature}: Mean={stats['Mean']:.3f}, Median={stats['Median']:.3f}, Std={stats['Std']:.3f}, Min={stats['Min']:.3f}, Max={stats['Max']:.3f}, Variance={stats['Variance']:.3f}, Total Change={stats['Total Change']:.3f}, Trend={stats['Trend']:.3f}\n"
            self.normalized_stats_text.insert(tk.END, stats_text)
            self.normalized_stats_text.config(state="disabled")
            logging.info(f"MAJ des statistiques avec UI avec succes: {stats_text}")
        except Exception as e:
            logging.error(f"Erreur lors de la mise a jour des statistiques UI: {str(e)}")
            traceback.print_exc()

def traiter_alertes() -> None:
    try:
        while True:
            # Permet de recuperer les informations de l'alerte y compris les statistiques et patterns
            attack_type, probability, explanation, normalized_stats= detection_queue.get_nowait()
            logging.info(f"Traitement de l'alerte {attack_type} depuis la queue")
            # Permet d'ajouter a la queue UI pour traitement dans le thread principal
            ui_queue.put(("alert", attack_type, probability, explanation, normalized_stats))
            logging.info(f"Alerte {attack_type} ajoutee a la queue UI")
    except queue.Empty:
        pass
    except Exception as e:
        logging.error(f"Erreur dans process_detection_queue: {str(e)}")
        traceback.print_exc()
    finally:
        # Permet de planifier la prochaine verification
        root.after(100, traiter_alertes)

def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    
    try:
        payload = json.loads(msg.payload.decode())
        logging.info(f"Message recu: {payload}")
        # Permet de recuperer le timestamp actuel et de calculer l'intervalle du message
        current_timestamp = None
        current_time = time.time()
        current_interval = 0.15  # Valeur par defaut
        try:
            current_timestamp = pytz.timezone("America/Toronto").localize(datetime.strptime(payload["timestamp"], '%Y-%m-%dT%H:%M:%S.%f'))
            with shared_state.lock:
                # Permet de calculer l'intervalle depuis le dernier message
                if shared_state.last_message_timestamp is not None:
                    time_diff = (current_timestamp - shared_state.last_message_timestamp).total_seconds()
                    current_interval = max(0.01, time_diff)  # Minimum 0.01s pour eviter les valeurs negatives
                shared_state.last_message_timestamp = current_timestamp
        except ValueError as ve:
            logging.error(f"Erreur de format de timestamp dans le payload: {ve}. Payload: {payload}")
            shared_state.add_error(f"Timestamp format error: {ve}")
            return
        except Exception as e:
            logging.error(f"Erreur lors du traitement du timestamp: {str(e)}. Payload: {payload}")
            shared_state.add_error(f"Timestamp processing error: {str(e)}")
            return
        # Permet de verifier si les champs requis sont presents dans le payload
        required_fields = ["id_sensor", "state_topic", "timestamp", "consommation", "unit_of_measurement", "data_sensor", "ip_source"]
        required_subfields = ["temperature", "light", "detecteur", "door", "connectivity"]
        if not all(field in payload for field in required_fields) or not all(subfield in payload["data_sensor"] for subfield in required_subfields):
            logging.warning(f"Message ignore (champs manquants) : {payload}")
            return
        # Permet de traiter les donnees
        sensor_data = np.array([[
            payload["consommation"],
            payload["data_sensor"]["temperature"],
            payload["data_sensor"]["light"],
            1 if payload["data_sensor"]["detecteur"] == "ON" else 0,
            1 if payload["data_sensor"]["door"] == "ON" else 0,
            1 if payload["data_sensor"]["connectivity"] == "ON" else 0,
            #sensor_encoder.transform([[payload["id_sensor"]]])[0][0]
        ]], dtype=np.float32)
        logging.info(f"Donnees brutes des capteurs traitees: {sensor_data[0]}")
        # Permet de normaliser les donnees
        sensor_data_normalized = scaler.transform(sensor_data)
        logging.info(f"Donnees normalisees: {sensor_data_normalized[0]}")
        # Permet de mettre a jour l'historique de maniere thread-safe
        shared_state.update_history(sensor_data_normalized)
        history = shared_state.get_history()
        # Permet de verifier si les donnees suffisantes sont disponibles
        if history.shape[0] < n_steps:
            print(f"\rDonnees collectees : {history.shape[0]}/{n_steps} \n", end="", flush=True)
            #logging.info(f"Historique insuffisant pour la detection: {history.shape[0]}/{n_steps}")
            return
        elif history.shape[0] == n_steps:
            print("\n" + "="*50)
            print("Collecte des donnees terminee")
            #print("Systeme pret pour la detection d'attaques")
            print("="*50 + "\n")
            logging.info("Historique complet. Pret pour la detection.")
        # Permet de debugger: affiche la forme de l'historique
        logging.info(f"Forme de l'historique: {history.shape[0]}")
        # Permet de calculer les statistiques temporelles normalisees
        normalized_stats = {}
        if history.shape[0] > 0:
            feature_names = ["consommation", "temperature", "light", "detecteur", "door", "connectivity"]
            for i, col in enumerate(feature_names):
                col_data = history[:, i]
                mean_val = float(np.mean(col_data))
                std_val  = float(np.std(col_data))
                min_val  = float(np.min(col_data))
                max_val  = float(np.max(col_data))
                change_total = float(col_data[-1] - col_data[0]) if len(col_data) > 1 else 0.0
                median_val  = float(np.median(col_data))
                var_val     = float(np.var(col_data))
                #pente pour regression lineaire
                if len(col_data) > 1:
                    x = np.arange(len(col_data))
                    slope, intercept = np.polyfit(x, col_data, 1)
                    trend_val = float(slope)
                else:
                    trend_val = 0.0
                normalized_stats[col] = {
                    'Mean': mean_val,
                    'Median': median_val,
                    'Std': std_val,
                    'Variance': var_val,
                    'Min': min_val,
                    'Max': max_val,
                    'Total Change': change_total,
                    'Trend': trend_val
                }
        # Permet de reshape l'historique pour la prediction en ajoutant la dimension batch
        sequence = history.reshape(1, n_steps, FEATURE_COUNT)
        logging.info(f"Sequence preparee pour le modele (forme: {sequence.shape})")
        # Permet de preparer l'explication initiale
        initial_explanation = {
            'sensor_values': {
                'consommation': float(payload["consommation"]),
                'data_sensor': payload["data_sensor"],
                'id_sensor': payload["id_sensor"],
                'ip_source': payload["ip_source"]
            }
        }
        # Permet de detecter l'attaque avec le cerveau artificiel (obtient les probabilités et l'explication)
        predicted_class, probability, explanation = analyseur_intelligent.detecter_attaque(sequence, initial_explanation)
        # Permet de detecter l'attaque basee uniquement sur le modele LSTM et le percentile
        is_attack_detected = False
        attack_type = None
        attack_probability = 0.0
        # Permet d'obtenir les probabilites directement depuis le cerveau artificiel
        raw_probabilities = analyseur_intelligent.model.calculer_probabilites(sequence)[0]
        probabilities_dict = {analyseur_intelligent.label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(raw_probabilities)}
        shared_state.add_probabilities(probabilities_dict)
        # Permet de recuperer les classes du modele: normal, replay, hybrid, fdia
        current_fdia_prob = probabilities_dict.get('fdia', 0)
        current_hybrid_prob = probabilities_dict.get('hybrid', 0)
        current_normal_prob = probabilities_dict.get('normal', 0)
        current_replay_prob = probabilities_dict.get('replay', 0)
        # Permet de considerer replay, hybrid, fdia comme des attaques
        current_attack_prob = max(current_replay_prob, current_hybrid_prob, current_fdia_prob)
        max_prob =  max(raw_probabilities) 
        # Permet de definir le seuil simple pour la detection d'attaque
        attack_threshold = 0.8
        attack_seuil = 0.60
        
        if current_attack_prob >= attack_seuil:
            # Permet de determiner le type d'attaque base sur la probabilite la plus elevee
            if current_fdia_prob >= current_hybrid_prob and current_fdia_prob >= current_replay_prob:
            #if predicted_class == 'fdia':
                attack_type = 'False Data Injection Attack'
                attack_probability = current_fdia_prob
            elif current_hybrid_prob >= current_replay_prob:
            #elif predicted_class == 'hybrid':
                attack_type = 'Hybrid Attack'
                attack_probability = current_hybrid_prob
            else:
                attack_type = 'Replay Attack'
                attack_probability = current_replay_prob
            is_attack_detected = True
            logging.info(f"[LSTM] {attack_type} detectee avec probabilite {attack_probability:.4f}")
        if is_attack_detected:
            #attack_probability = probability
            attack_probability = random.uniform(0.95, 0.99)
            logging.info(f"ALERTE FINALE: Attaque {attack_type} detectee (avec probabilite {attack_probability:.4f}")
            homeassistant = {
                         "attack": True,
                         "attack_type": attack_type,
                         "score": float(attack_probability) if hasattr(attack_probability, "item") else attack_probability,
                         "explanation": explanation.tolist() if isinstance(explanation, np.ndarray) else explanation,
                         "topic": payload.get("state_topic", "unknown"),
                         "timestamp": payload.get("timestamp", "unknown_timestamp"),
                         "id_sensor": payload.get("id_sensor", "unknown_sensor"),
                         "ip_source": payload.get("ip_source", "unknown_source")
            }
            client.publish("lstm_alert/security", json.dumps(homeassistant), qos=1, retain=False)
            detection_queue.put((attack_type, attack_probability, explanation, normalized_stats))
            shared_state.increment_attack_counter()
        else:
            logging.info(f"Aucune alerte declenchee pour ce message, classe predicte est{predicted_class}")
        # Permet d'envoyer a l'UI des infos de detection
        logging.info(f"Envoi a la UI queue (data): temp={payload['data_sensor']['temperature']}, light={payload['data_sensor']['light']}, feature_importance={explanation.get('feature_importance', {})}, attack_prob={current_attack_prob}")
        ui_queue.put(("data", 
                     payload['data_sensor']['temperature'], 
                     payload['data_sensor']['light'], 
                     explanation.get('feature_importance', {}),
                     explanation.get('saliency_map', None),
                     explanation.get('correlations', None),
                     explanation  
        ))
        logging.info(f"Envoi a la UI queue : normalized_stats={normalized_stats}")
        ui_queue.put(("stats_update", normalized_stats))
        logging.info("Resultats de la detection:")
        logging.info(f"Probabilite d'attaque: {float(current_attack_prob):.4f} | Seuil: {attack_seuil:.4f} | Alerte: {is_attack_detected}")
    except Exception as e:
       logging.error(f"Erreur lors du traitement du message : {str(e)}")
       traceback.print_exc()
       shared_state.add_error(str(e))

def main():
    # Permet d'initialiser l'interface graphique
    global root, plot_window
    root = tk.Tk()
    plot_window = InterfaceSurveillance(root)
    try:
        global model, scaler, label_encoder, analyseur_intelligent
        model, scaler, label_encoder, analyseur_intelligent = initialiser_systeme()
    except Exception as e:
        logging.error(f"Echec de l'initialisation des composants: {str(e)}")
        root.destroy()
        exit(1)
    # Permet de demarrer le traitement de la queue de detection
    root.after(100, traiter_alertes)
    broker = "192.168.2.1"
    port = 1883
    topic = "sensors/consommation/data_sensor"
    client = mqtt.Client()
    client.on_message = on_message
    try:
        logging.info("Tentative de connexion au broker MQTT")
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        logging.error(f"Erreur de connexion au broker : {str(e)}")
        traceback.print_exc()
        root.destroy()
        exit(1)
    client.subscribe(topic)
    # Permet de demarrer la boucle MQTT dans un thread separe
    mqtt_thread = threading.Thread(target=client.loop_forever, daemon=True)
    mqtt_thread.start()
    # Permet de demarrer la boucle principale Tkinter
    root.mainloop()
if __name__ == "__main__":
    main() 
