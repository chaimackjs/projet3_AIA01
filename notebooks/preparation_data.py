# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration de l'affichage
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath):
    """
    Charge les données depuis un fichier CSV
    
    Parameters:
        filepath (str): Chemin vers le fichier CSV
    
    Returns:
        DataFrame: Données chargées
    """
    data = pd.read_csv(filepath)
    return data



def analyze_data_quality(data):
    """
    Analyse la qualité des données (valeurs manquantes, doublons, types)
    
    Parameters:
        data (DataFrame): Données à analyser
    """
    print("\n" + "="*80)
    print("ANALYSE DE LA QUALITÉ DES DONNÉES")
    print("="*80)
    
    # Types de données
    print("\nTypes de données par colonne:")
    print(data.dtypes)
    
    # Valeurs manquantes
    print("\nAnalyse des valeurs manquantes:")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Valeurs_Manquantes': missing_values,
        'Pourcentage': missing_percentage
    })
    
    missing_df = missing_df[missing_df['Valeurs_Manquantes'] > 0]
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("Aucune valeur manquante détectée")
    
    # Doublons
    print(f"\nNombre de doublons: {data.duplicated().sum()}")
    
    return missing_df


# Programme principal
def main():
    """
    Fonction principale pour exécuter toutes les étapes de préparation des données
    """
    # Chargement des données
    print("="*80)
    print("PRÉPARATION DES DONNÉES - TELCO CUSTOMER CHURN")
    print("="*80)
    
    data = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"\nDonnées chargées: {data.shape[0]} lignes, {data.shape[1]} colonnes")


    # Analyse de la qualité des données
    missing_df = analyze_data_quality(data)
    

if __name__ == "__main__":
    main()