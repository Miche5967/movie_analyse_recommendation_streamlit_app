# Système de recommandation de films dans une application Streamlit (projet pédagogique de formation)

## A propos

Projet pédagogique de formation réalisé au cours de la formation *Data Analyst* de la Wild Code School.  
Ce projet a pour but de créer un système de recommandation de films à partir de datasets disponibles sur le site imdb.com (https://datasets.imdbws.com/).  
- Les fichiers présents dans ce repository sont ceux utilisés par l'application *Streamlit* . Le script de l'application Streamlit est
- Les fichies utilisés pour réaliser toutes les étapes de traitement des données récupérées depuis le site IMDb sont disponibles dans un autre repository : https://github.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films.

## Lien vers l'application

https://miche5967-movie-analyse-recommendation--movie-app-script-cwtn73.streamlit.app/

## Descriptif des fichiers présents

### movie_app_script.py
Script de l'application. Ce fichier contient l'ensemble du script utilisé par Streamlit pour afficher l'application.
- Fonctions de chargement et de traitement des données depuis des fichiers csv déjà traités présents sur le repository https://github.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films
- Widgets pour interagir : bouton radio pour choisir la page, checkbox pour choisir les genres, onglets pour naviguer entre les pages des graphiques
- Visualisations graphiques pour le tableau de bord de l'application : données sur les films par genre, sur les acteurs et réalisateurs
- Système de recommandation de films avec une zone de saisie.

### requirements.txt
Fichier contenant les versions des packages utilisés par l'application.

Il y a aussi un logo en png.

## Fabriqué avec

### Langages et librairies
- Python
  - Pandas,
  - Plotly Express,
  - Scikit-Learn,
  - Streamlit

### Logiciels
- Jupyter Notebook
- Sublime Text

## Auteurs

* **Julien Michaut** _alias_ [@Miche5967](https://github.com/Miche5967)
* **Geoffrey Castel**
* **Mathias Nieuwjaer**

