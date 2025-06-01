# DANN - Diagnostic de Défauts

Réseau de neurones à adaptation de domaine pour transférer les connaissances des données de simulation vers les données réelles.

## Fichiers

- `models.py` : Architecture du réseau DANN
- `Dann.py` : Script d'entraînement

## Installation

```bash
pip install torch numpy scikit-learn
```

## Utilisation

```bash
python Dann.py --simu-dir "../training_csv" --real-dir "../training_csv"
```

### Paramètres principaux

- `--simu-dir` : Dossier des données de simulation (requis)
- `--real-dir` : Dossier des données réelles (requis) 
- `--epochs` : Nombre d'époques (défaut: 50)
- `--output` : Fichier de sauvegarde du modèle


## Structure des données

```
data_directory/
├── Healthy/
├── Motor_1_Stuck/
├── Motor_1_Steady_state_error/
└── ...
```

Chaque fichier CSV : 1000 lignes × 6 colonnes.
