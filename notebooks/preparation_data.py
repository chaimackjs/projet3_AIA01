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


def perform_eda(data):
    """
    Effectue une analyse exploratoire des données
    
    Parameters:
        data (DataFrame): Données à analyser
    """
    print("\n" + "="*80)
    print("ANALYSE STATISTIQUE DESCRIPTIVE")
    print("="*80)
    
    # Séparation des variables par type
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nColonnes numériques ({len(numerical_cols)}): {numerical_cols}")
    print(f"Colonnes catégorielles ({len(categorical_cols)}): {categorical_cols}")
    
    # Statistiques descriptives
    print("\nStatistiques des variables numériques:")
    print(data[numerical_cols].describe())
    
    # Distribution de la variable cible
    print("\nDistribution de la variable cible (Churn):")
    churn_distribution = data['Churn'].value_counts()
    churn_percentage = data['Churn'].value_counts(normalize=True) * 100
    print(pd.DataFrame({
        'Count': churn_distribution,
        'Percentage': churn_percentage
    }))
    
    return numerical_cols, categorical_cols, churn_distribution



def visualize_target_distribution(churn_distribution):
    """
    Visualise la distribution de la variable cible
    
    Parameters:
        churn_distribution: Distribution du churn
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique en barres
    axes[0].bar(churn_distribution.index, churn_distribution.values, 
                color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Distribution du Churn', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Nombre de clients')
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate(churn_distribution.values):
        axes[0].text(i, v + 50, str(v), ha='center')
    
    # Graphique en secteurs
    axes[1].pie(churn_distribution.values, labels=churn_distribution.index, 
                autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Proportion du Churn', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualize_numerical_features(data):
    """
    Visualise les variables numériques en fonction du churn
    
    Parameters:
        data (DataFrame): Données à visualiser
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Tenure
    axes[0].boxplot([data[data['Churn']=='No']['tenure'].dropna(), 
                     data[data['Churn']=='Yes']['tenure'].dropna()],
                    labels=['No Churn', 'Churn'])
    axes[0].set_title('Tenure vs Churn')
    axes[0].set_ylabel('Tenure (mois)')
    
    # MonthlyCharges
    axes[1].boxplot([data[data['Churn']=='No']['MonthlyCharges'].dropna(), 
                     data[data['Churn']=='Yes']['MonthlyCharges'].dropna()],
                    labels=['No Churn', 'Churn'])
    axes[1].set_title('Monthly Charges vs Churn')
    axes[1].set_ylabel('Charges mensuelles ($)')
    
    # TotalCharges
    axes[2].boxplot([data[data['Churn']=='No']['TotalCharges'].dropna(), 
                     data[data['Churn']=='Yes']['TotalCharges'].dropna()],
                    labels=['No Churn', 'Churn'])
    axes[2].set_title('Total Charges vs Churn')
    axes[2].set_ylabel('Charges totales ($)')
    
    plt.tight_layout()
    plt.show()


def visualize_categorical_features(data):
    """
    Visualise le taux de churn pour les variables catégorielles importantes
    
    Parameters:
        data (DataFrame): Données à visualiser
    """
    important_categorical = ['Contract', 'InternetService', 'PaymentMethod', 
                            'TechSupport', 'OnlineSecurity']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(important_categorical):
        if idx < len(important_categorical):
            churn_rates = data.groupby(col)['Churn'].apply(
                lambda x: (x=='Yes').sum() / len(x) * 100
            )
            axes[idx].bar(range(len(churn_rates)), churn_rates.values, color='coral')
            axes[idx].set_title(f'Taux de Churn par {col}')
            axes[idx].set_xticks(range(len(churn_rates)))
            axes[idx].set_xticklabels(churn_rates.index, rotation=45, ha='right')
            axes[idx].set_ylabel('Taux de Churn (%)')
            
            # Ajout des valeurs sur les barres
            for i, v in enumerate(churn_rates.values):
                axes[idx].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
    
    # Suppression du subplot vide
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show()


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

    # Analyse exploratoire
    numerical_cols, categorical_cols, churn_distribution = perform_eda(data)

    # Visualisations
    visualize_target_distribution(churn_distribution)
    visualize_numerical_features(data)
    visualize_categorical_features(data)     


if __name__ == "__main__":
    main()