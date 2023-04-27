import mlflow
import sql
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle

from sql import get_mortality, get_bleached_data, get_growth_data

mainzones = ['Water Villas (LG)', 'Channel (KH)', 'Blue Holes (LG)', 'Parrot Reef (LG)', 'Water Villas (KH)', 'House Reef (KH)', 'Blu (LG)', 'Blu Deep (LG)', 'Dive Site (LG)', 'Coral Trail (LG)', 'Anchor Point (LG)', 'Al Barakat (LG)']

play = False

if play == True:
    # Téléchargement et prétraitement des modèles de mortalité
    survived, dead = get_mortality()
    with open('survived_dead.pkl', 'wb') as f:
        pickle.dump((survived, dead), f)

else:
    with open('survived_dead.pkl', 'rb') as f:
        survived, dead = pickle.load(f)
    roll_window = 120
    mortality_dict = {}
    bleached_dict = {}
    growth_dict = {}
    lines = {}
    fig, axs = plt.subplots(2, sharex = True, figsize = [13,8])


    # Calcul du taux de mortalité pour chaque zone
    for z in survived['Zone'].unique():
        zone_survived_counts = survived['Median'].value_counts()
        zone_dead_counts = dead['Median'].value_counts()
        mortality_dict[z] = zone_dead_counts / (zone_dead_counts + zone_survived_counts)
        mortality_dict[z] = mortality_dict[z].sort_index().rolling(str(roll_window) + 'd').sum() / roll_window * 100
        lines[z], = axs[0].plot_date(mortality_dict[z].index, mortality_dict[z], '-')

        # Calcul du taux de blanchissement pour chaque zone
        zone_data = get_bleached_data()
        zone_data = zone_data[zone_data['Zone'] == z]
        bleached_dict[z] = zone_data.groupby(['ObsDate'])['Outcome'].apply(lambda x: (x == 'Bleached Corail').sum() / len(x) * 100)

        # Calcul du taux de croissance pour chaque zone
        growth_data = get_growth_data()
        zone_growth = growth_data[growth_data['Zone'] == z]
        live_coral_data = zone_growth[(zone_growth['Type'] == 'Acropora') | (zone_growth['Type'] == 'Pocillopora')]
        live_coral_data = live_coral_data[['avgrowth']]
        if len(live_coral_data) > 0:
            growth_rate = live_coral_data.sort_index().rolling(int(roll_window)).mean() - 1
            growth_rate['Zone'] = zone_growth['Zone']
            growth_rate_mean = growth_rate.groupby('Zone')['avgrowth'].mean() # calcul des moyennes par zone
            growth_dict[z] = growth_rate_mean
            for mz in growth_rate_mean.index:
                mz_data = growth_rate[growth_rate['Zone'] == mz]
                lines[mz], = axs[0].plot_date(mz_data.index, mz_data['avgrowth'], '-', label=mz)
        else:
            print('No live coral data found for zone', z)

        # Création imputer pour remplacer les valeurs manquantes par la moyenne
        imputer = SimpleImputer(strategy='mean')

        # Entrainement du modèle de régression linéaire pour le taux de mortalité
        print(mortality_dict[z].values)
        print(bleached_dict[z].values)
        print(growth_dict.keys())
        print(growth_dict[z].values)

        X = pd.to_numeric(mortality_dict[z].index).values.reshape(-1,1)
        y = mortality_dict[z].values.reshape(-1,1)
        model = LinearRegression().fit(X,y)

        # Entrainement du modèle de régression linéaire pour le taux de blanchissement
        bleached_dict[z].index = pd.to_datetime(bleached_dict[z].index)
        X_bleach = pd.to_numeric(bleached_dict[z].index, errors = 'coerce').values.reshape(-1,1)
        y_bleach = bleached_dict[z].values.reshape(-1,1)
        if len(X_bleach) > 0: 
            model_bleach = LinearRegression().fit(X_bleach, y_bleach)
        else:
            print('Le tableau X_bleach est vide')

        # Entrainement du modèle de régression linéaire pour le taux de croissance
        growth_data = growth_data.fillna(growth_data.mean(numeric_only=True))# remplacer les valeurs manquantes par la moyenne de la colonne
        X_growth = pd.to_numeric(growth_dict[z].index, errors = 'coerce').values.reshape(-1,1)
        X_growth = imputer.fit_transform(X_growth)
        print(X_growth.shape)
        y_growth = growth_dict[z].values.reshape(-1,1)
        model_growth = LinearRegression().fit(X_growth, y_growth)

        # Définition des différents suivis
        experiment_mortality = 'Taux de mortalité'
        experiment_bleached = 'Taux de blanchissement'
        experiment_growth = 'Taux de croissance'

        # Définir les dates futures
        start_date = pd.Timestamp(datetime.now().date() + pd.Timedelta(days=1))
        end_date = pd.Timestamp(datetime.now().date() + pd.Timedelta(days=30))
        future_dates = pd.date_range(start_date, end_date, freq='D')

        # Pour chaque date future, calculer les caractéristiques nécessaires et faire des prédictions
        for date in future_dates:
        # Calculer la caractéristique pour la date future
            feature_value = (date - pd.Timestamp(datetime.now().date())).days

            # Faire une prédiction en utilisant le modèle entrainé
            prediction = model.predict([[feature_value]])

        # Début traçabilité Mlflow pour chaque expérience

        # Suivi taux de mortalité
        with mlflow.start_run(run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")) as run_mortality:
            mlflow.set_experiment(experiment_mortality)
            mlflow.log_param("Zone", z)
            mlflow.log_param("Date", date.strftime("%Y-%m-%d"))

            # Enregistrement des métriques
            mlflow.log_metric('R²', model.score(X,y))
            mlflow.log_metric("Prediction", prediction[0])

            # Enregistrement de la figure de mortalité de chaque mainzones
            fig, ax = plt.subplots(figsize=(10, 6))
            x = pd.to_datetime(mortality_dict[z].index).to_pydatetime()
            ax.plot(x, np.array(mortality_dict[z].values), label = z)
            ax.set_xlabel("Temps")
            ax.set_ylabel("Taux de mortalité (%)")
            ax.set_title(f"Taux de mortalité dans la zone {z}")
            ax.text(0.95, 0.95, f"R²={model.score(X, y):.2f}", transform=ax.transAxes, ha="right", va="top")
            fig.autofmt_xdate() # éviter que les informations se chevauchent
            fig.savefig(f"mortality_{z}.png")
            mlflow.log_artifact(f"mortality_{z}.png")
            plt.close(fig)

        #Suivi taux de blanchissement
        with mlflow.start_run(run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")) as run_bleached:
            mlflow.set_experiment(experiment_bleached)
            mlflow.log_param("Zone", z)

            # Enregistrement des métriques
            mlflow.log_metric('R²', model_bleach.score(X_bleach, y_bleach))

            # Enregistrement de la figure de blanchissement de chaque mainzones
            fig, ax = plt.subplots(figsize=(10, 6))
            x = pd.to_datetime(bleached_dict[z].index).to_pydatetime()
            ax.plot(x, np.array(bleached_dict[z].values), label = z)
            ax.set_xlabel("Temps")
            ax.set_ylabel("Taux de blanchissement (%)")
            ax.set_title(f"Taux de blanchissement dans la zone {z}")
            ax.text(0.95, 0.95, f"R²={model_bleach.score(X_bleach, y_bleach):.2f}", transform=ax.transAxes, ha="right", va="top")
            fig.autofmt_xdate() # éviter que les informations se chevauchent
            fig.savefig(f"bleached_{z}.png")
            mlflow.log_artifact(f"bleached_{z}.png")
            plt.close(fig)

        # Suivi taux de croissance
        with mlflow.start_run(run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")) as run_growth:
            mlflow.set_experiment(experiment_growth)
            mlflow.log_param("Zone", z)

            # Enregistrement des métriques
            mlflow.log_metric('R²', model_growth.score(X_growth, y_growth))

            # Enregistrement de la figure de blanchissement de chaque mainzones
            fig, ax = plt.subplots(figsize=(10, 6))
            x = pd.to_datetime(growth_dict[z].index).to_pydatetime()
            ax.plot(x, np.array(growth_dict[z].values), label = z)
            ax.set_xlabel("Temps")
            ax.set_ylabel("Taux de croissance (%)")
            ax.set_title(f"Taux de croissance dans la zone {z}")
            ax.text(0.95, 0.95, f"R²={model_growth.score(X_growth, y_growth):.2f}", transform=ax.transAxes, ha="right", va="top")
            fig.autofmt_xdate() # éviter que les informations se chevauchent
            fig.savefig(f"growth_{z}.png")
            mlflow.log_artifact(f"growth_{z}.png")
            plt.close(fig)

        mlflow.end_run()