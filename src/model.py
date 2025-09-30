# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import time
import os

# Bibliothèques de modélisation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (cross_val_score, GridSearchCV, 
                                   RandomizedSearchCV, StratifiedKFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           roc_curve, classification_report)

# Gestion du déséquilibre des classes
from imblearn.over_sampling import SMOTE

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_prepared_data(data_dir='../data/processed'):
    """
    Charge les données préparées
    
    Parameters:
        data_dir (str): Répertoire contenant les données
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\n" + "="*80)
    print("CHARGEMENT DES DONNÉES")
    print("="*80)
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    print(f"Données chargées:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - Taux de churn train: {y_train.mean():.2%}")
    print(f"  - Taux de churn test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):
    """
    Équilibre les classes avec SMOTE
    
    Parameters:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
    
    Returns:
        tuple: X_train_balanced, y_train_balanced
    """
    print("\n" + "="*80)
    print("ÉQUILIBRAGE DES CLASSES (SMOTE)")
    print("="*80)
    
    print(f"Avant SMOTE:")
    print(f"  - Taille: {len(y_train)}")
    print(f"  - Proportion classe 1: {y_train.mean():.2%}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nAprès SMOTE:")
    print(f"  - Taille: {len(y_train_balanced)}")
    print(f"  - Proportion classe 1: {y_train_balanced.mean():.2%}")
    
    return X_train_balanced, y_train_balanced


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Évalue un modèle et affiche les métriques de performance
    
    Parameters:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        model_name (str): Nom du modèle
    
    Returns:
        dict: Métriques de performance
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"PERFORMANCE: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Matrice de confusion
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                ax=axes[0])
    axes[0].set_title(f'Matrice de Confusion - {model_name}')
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_xlabel('Classe prédite')
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'Courbe ROC - {model_name}')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Retour des métriques
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne et optimise un modèle de régression logistique
    
    Parameters:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
    
    Returns:
        tuple: (modèle de base, modèle optimisé, résultats)
    """
    print("\n" + "="*80)
    print("MODÈLE: RÉGRESSION LOGISTIQUE")
    print("="*80)
    
    # Entraînement du modèle de base
    print("\nEntraînement du modèle de base...")
    start_time = time.time()
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    print(f"Temps d'entraînement: {time.time() - start_time:.2f} secondes")
    
    # Cross-validation
    print("\nCross-validation (5-fold):")
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Score moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Évaluation du modèle de base
    lr_results = evaluate_model(lr_model, X_test, y_test, "Régression Logistique")
    
    # Optimisation des hyperparamètres
    print("\nOptimisation des hyperparamètres...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    
    # Évaluation du modèle optimisé
    lr_best_results = evaluate_model(
        grid_search.best_estimator_, X_test, y_test, 
        "Régression Logistique (Optimisée)"
    )
    
    return lr_model, grid_search.best_estimator_, (lr_results, lr_best_results)

def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Entraîne et optimise un arbre de décision
    
    Parameters:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
    
    Returns:
        tuple: (modèle de base, modèle optimisé, résultats)
    """
    print("\n" + "="*80)
    print("MODÈLE: ARBRE DE DÉCISION")
    print("="*80)
    
    # Entraînement du modèle de base
    print("\nEntraînement du modèle de base...")
    start_time = time.time()
    
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    
    print(f"Temps d'entraînement: {time.time() - start_time:.2f} secondes")
    
    # Cross-validation
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Visualisation de l'arbre
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, max_depth=3, feature_names=X_train.columns,
              class_names=['No Churn', 'Churn'], filled=True, 
              rounded=True, fontsize=10)
    plt.title("Arbre de Décision - Visualisation (profondeur 3)")
    plt.show()
    
    # Évaluation - CETTE LIGNE ÉTAIT MANQUANTE
    dt_results = evaluate_model(dt_model, X_test, y_test, "Arbre de Décision")
    
    # Optimisation des hyperparamètres
    print("\nOptimisation des hyperparamètres...")
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    
    # Évaluation du modèle optimisé
    dt_best_results = evaluate_model(
        grid_search.best_estimator_, X_test, y_test, 
        "Arbre de Décision (Optimisé)"
    )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': grid_search.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print("\nTop 10 Features importantes:")
    print(feature_importance)
    
    return dt_model, grid_search.best_estimator_, (dt_results, dt_best_results)

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Entraîne et optimise un Random Forest
    
    Parameters:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
    
    Returns:
        tuple: (modèle de base, modèle optimisé, résultats)
    """
    print("\n" + "="*80)
    print("MODÈLE: RANDOM FOREST")
    print("="*80)
    
    # Entraînement du modèle de base
    print("\nEntraînement du modèle de base...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print(f"Temps d'entraînement: {time.time() - start_time:.2f} secondes")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Évaluation
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Optimisation avec RandomizedSearchCV
    print("\nOptimisation des hyperparamètres...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {random_search.best_params_}")
    print(f"Meilleur score CV: {random_search.best_score_:.4f}")
    
    # Évaluation du modèle optimisé
    rf_best_results = evaluate_model(
        random_search.best_estimator_, X_test, y_test, 
        "Random Forest (Optimisé)"
    )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': random_search.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualisation des top 15 features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values, color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 15 Features - Random Forest')
    plt.gca().invert_yaxis()
    
    # Ajout des valeurs
    for i, v in enumerate(top_features['importance'].values):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Features importantes:")
    print(feature_importance.head(10))
    
    return rf_model, random_search.best_estimator_, (rf_results, rf_best_results)


def compare_models(results_list):
    """
    Compare les performances de tous les modèles
    
    Parameters:
        results_list (list): Liste des résultats de chaque modèle
    
    Returns:
        DataFrame: Tableau comparatif
    """
    print("\n" + "="*80)
    print("COMPARAISON DES MODÈLES")
    print("="*80)
    
    # Création du DataFrame de comparaison
    results_comparison = pd.DataFrame(results_list)
    
    # Affichage du tableau
    print("\nTableau comparatif des performances:")
    print(results_comparison.to_string(index=False))
    
    # Identification du meilleur modèle
    best_model_idx = results_comparison['roc_auc'].idxmax()
    best_model = results_comparison.loc[best_model_idx]
    print(f"\nMeilleur modèle: {best_model['model_name']}")
    print(f"ROC AUC: {best_model['roc_auc']:.4f}")
    
    # Visualisation comparative
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx//3, idx%3]
        bars = ax.bar(range(len(results_comparison)), 
                      results_comparison[metric], 
                      color=colors[idx % len(colors)])
        ax.set_title(f'Comparaison - {metric.upper()}', fontweight='bold')
        ax.set_xticks(range(len(results_comparison)))
        ax.set_xticklabels(results_comparison['model_name'], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, results_comparison[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Suppression du subplot vide
    fig.delaxes(axes[1, 2])
    
    plt.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return results_comparison


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Trace les courbes ROC de tous les modèles sur un même graphique
    
    Parameters:
        models_dict (dict): Dictionnaire des modèles
        X_test: Features de test
        y_test: Labels de test
    """
    print("\nGénération des courbes ROC comparatives...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, (name, model) in enumerate(models_dict.items()):
        # Prédictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    # Ligne de référence
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Courbes ROC - Comparaison des modèles', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_best_model(model, model_name, output_dir='models'):
    """
    Sauvegarde le meilleur modèle
    
    Parameters:
        model: Modèle à sauvegarder
        model_name (str): Nom du modèle
        output_dir (str): Répertoire de sauvegarde
    """
    # Création du répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Nom du fichier
    filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}.pkl"
    
    # Sauvegarde
    joblib.dump(model, filename)
    print(f"\nModèle sauvegardé: {filename}")


# Programme principal
def main():
    """
    Fonction principale pour exécuter toute la pipeline de modélisation
    """
    print("="*80)
    print("MODÉLISATION - PRÉDICTION DU CHURN CLIENT")
    print("="*80)
    
    # Chargement des données
    X_train, X_test, y_train, y_test = load_prepared_data()
    
    # Équilibrage des classes
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    
    # Liste pour stocker tous les résultats
    all_results = []
    
    # Entraînement de la régression logistique
    lr_model, lr_best, (lr_results, lr_best_results) = train_logistic_regression(
        X_train_balanced, y_train_balanced, X_test, y_test
    )
    all_results.extend([lr_results, lr_best_results])
    
    # Entraînement de l'arbre de décision
    dt_model, dt_best, (dt_results, dt_best_results) = train_decision_tree(
        X_train_balanced, y_train_balanced, X_test, y_test
    )
    all_results.extend([dt_results, dt_best_results])
    
    # Entraînement du Random Forest
    rf_model, rf_best, (rf_results, rf_best_results) = train_random_forest(
        X_train_balanced, y_train_balanced, X_test, y_test
    )
    all_results.extend([rf_results, rf_best_results])
    
    # Comparaison des modèles
    comparison_df = compare_models(all_results)
    
    # Dictionnaire des modèles optimisés
    models_dict = {
        'Régression Logistique': lr_best,
        'Arbre de Décision': dt_best,
        'Random Forest': rf_best
    }
    
    # Courbes ROC comparatives
    plot_roc_curves(models_dict, X_test, y_test)
    
    # Identification et sauvegarde du meilleur modèle
    best_idx = comparison_df['roc_auc'].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'model_name']
    
    # Sélection du modèle correspondant
    if 'Random Forest' in best_model_name:
        best_model = rf_best
    elif 'Arbre' in best_model_name:
        best_model = dt_best
    else:
        best_model = lr_best
    
    # Sauvegarde
    save_best_model(best_model, best_model_name)
    
    print("\n" + "="*80)
    print("MODÉLISATION TERMINÉE")
    print("="*80)
    print(f"Meilleur modèle: {best_model_name}")
    print(f"Performance (ROC AUC): {comparison_df.loc[best_idx, 'roc_auc']:.4f}")


if __name__ == "__main__":
    main()