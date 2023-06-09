# Imports des librairies
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Changement de la largeur de la page pour l'élargir
st.set_page_config(layout="wide")

# Titre de la page
st.markdown("<h1 style='text-align: center;'>Analyses et recommandations de films</h1>", unsafe_allow_html=True)

# Affichage (temporaire) du "session_state"
#"st.session.state (*juste pour voir, à supprimer ensuite*) :", st.session_state

# Type de chargement des données
# Données déjà "traitées", chargées depuis github : True
# Sinon, depuis IMDb ou en local : False
data_loading_type_from_github = True

@st.cache_data
def load_and_process_title_akas_and_basics():
    '''
    Charge et traite les données de 2 fichiers lus sur le site d'IMDb, puis renvoie un DataFrame pandas
    contenant les données pertinentes.

    Returns:
    -------
    pandas.DataFrame
        DataFrame contenant les données filtrées et traitées des fichiers title.akas.tsv.gz
        et title.basics.tsv.gz.

    Notes:
    ------
    La fonction charge les fichiers title.akas.tsv.gz et title.basics.tsv.gz à partir d'URLs
    spécifiées du site IMDb. Les colonnes pertinentes sont sélectionnées pour chaque fichier,
    puis les données sont filtrées et traitées. Les deux DataFrames résultants sont fusionnés
    et les colonnes inutiles sont supprimées. Le DataFrame final contient les titres de films
    distribués en France à partir de 1980, avec les colonnes suivantes :
    - 'tconst': Identifiant unique du titre
    - 'title': Titre du film (en France)
    - 'startYear': Année de sortie du film
    - 'runtimeMinutes': Durée du film en minutes
    - 'genres': Genres du film
    '''

    # Définition des "chunks" pour optimiser la lecture
    chunksize = 600000
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_akas = ['titleId', 'title', 'region']

    # Initialisation d'un DataFrame pour la lecture du fichier title.akas
    df_title_akas_fr_trim = pd.DataFrame()
    
    # Lecture du fichier title.akas.tsv.gz avec des chunks
    #path_akas = r"https://datasets.imdbws.com/title.akas.tsv.gz"
    path_akas = r"C:/Données/d_ Wild Code School/d_ Projet 02/datasets/title_akas.tsv"
    df_chunks_akas = pd.read_csv(
    	path_akas, usecols = columns_to_include_akas,
    	dtype = {'titleId': 'string', 'title': 'string', 'region': 'string'}, delimiter = '\t',
    	chunksize = chunksize)
    
    for chunk in df_chunks_akas:
        # Filtrer et traiter chaque morceau
        
        # Filtre sur la colonne "region" pour ne garder que la valeur "FR"
        chunk = chunk[chunk["region"] == "FR"]

        # Suppression de la colonne "region"
        chunk.drop(columns = "region", inplace = True)

        # Concaténer le morceau traité au DataFrame final
        df_title_akas_fr_trim = pd.concat([df_title_akas_fr_trim, chunk])
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.basics
    columns_to_include_basics = ['tconst', 'titleType', 'startYear', 'runtimeMinutes', 'genres']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.basics
    df_title_basics_recent_years = pd.DataFrame()
    
    # Lecture du fichier title.basics.tsv.gz avec des chunks
    #path_basics = r"https://datasets.imdbws.com/title.basics.tsv.gz"
    path_basics = r"C:/Données/d_ Wild Code School/d_ Projet 02/datasets/title_basics.tsv"
    df_chunks = pd.read_csv(
        path_basics, usecols = columns_to_include_basics,
        dtype = {'tconst': 'string', 'titleType': 'string', 'startYear': 'string',
                 'runtimeMinutes': 'string', 'genres': 'string'},
        delimiter = '\t', low_memory = False, chunksize = chunksize)

    # Filtrer et traiter chaque morceau ("chunk")
    for chunk in df_chunks:
        
        # Suppression des lignes pour lesquelles la colonne "startYear" ne contient pas de nombre
        chunk = chunk[chunk["startYear"].str.isnumeric()]
        
        # Changement de type de données de la colonne "startYear", passage en type "integer"
        chunk["startYear"] = chunk["startYear"].astype("int32")
        
        # Filtre sur la colonne "startYear" pour ne garder que les années à partir de 1980
        chunk = chunk[chunk["startYear"] >= 1980]
        
        # Filtre sur la colonne "titleType" pour ne garder les titres de type "movie"
        chunk = chunk[chunk["titleType"] == "movie"]

        # Suppression des lignes où il n'y a pas de genre ("\N")
        chunk = chunk[chunk["runtimeMinutes"] != "\\N"]
        
        # Changement de type de données de la colonne "runtimeMinutes", passage en type "integer"
        chunk["runtimeMinutes"] = chunk["runtimeMinutes"].astype("int32")
        
        # Concaténer le morceau traité au DataFrame final
        df_title_basics_recent_years = pd.concat([df_title_basics_recent_years, chunk])
    
    # Fusion des deux DataFrame
    df_title_fr_recent_years = pd.merge(
        left = df_title_basics_recent_years, right = df_title_akas_fr_trim, how = 'inner',
        left_on = "tconst", right_on = "titleId")
    
    # Suppression des doublons
    df_movie_fr_recent_years_trim = df_title_fr_recent_years.copy()
    df_movie_fr_recent_years_trim.drop_duplicates(subset = 'tconst', inplace = True)
    
    # Suppression des colonnes inutiles
    # Colonne "titleType" devenue inutile car contient uniquement "movie"
    df_movie_fr_recent_years_trim.drop(columns = "titleType", inplace = True)
    # Colonne "titleId" car en doublon avec "tconst"
    df_movie_fr_recent_years_trim.drop(columns = "titleId", inplace = True)
    
    # Suppression des lignes où il n'y a pas de genre ("\N")
    df_movie_fr_recent_years_trim = df_movie_fr_recent_years_trim[
        df_movie_fr_recent_years_trim["genres"] != "\\N"]
    
    # Renvoi du DataFrame traité
    return df_movie_fr_recent_years_trim

@st.cache_data
def process_genres(df):
    '''
    Extrait les différents genres à partir du DataFrame donné, transforme la chaîne représentant les genres
    en liste, supprime certaines lignes en fonction des genres spécifiés, et renvoie un DataFrame des genres.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données traitées avec une colonne "genres" représentant les genres de chaque film.

    Returns:
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        Un tuple contenant deux DataFrames :
        - Le premier DataFrame est une copie du DataFrame d'entrée avec les modifications suivantes :
            - La colonne "genres" est convertie en listes de genres.
            - Les lignes contenant des genres spécifiés à supprimer sont supprimées.
        - Le deuxième DataFrame contient les genres et le nombre d'occurrences de chaque genre.

    Notes:
    ------
    La fonction extrait les différents genres présents dans la colonne "genres" du DataFrame donné.
    Elle transforme la chaîne de genres en une liste de genres. Ensuite, elle crée un dictionnaire
    qui associe chaque genre à son nombre d'occurrences. Un DataFrame des genres est ensuite créé à
    partir du dictionnaire, trié par le nombre d'occurrences décroissant. Enfin, la fonction supprime
    les lignes du DataFrame d'entrée qui contiennent certains genres spécifiés à supprimer, et renvoie
    le DataFrame modifié ainsi que le DataFrame des genres.
    '''

    # Création d'une copie du DataFrame et reset des index
    df_copy = df.copy()
    df_copy.reset_index(inplace = True, drop = True)
    df_copy["genres"] = df_copy["genres"].astype("object")

    # Initialisation d'un dictionnaire qui va servir à lister les genres présents, avec :
        # chaque clé est un genre
        # chaque valeur est la nombre d'occurences de ce genre
    dict_genres = {}

    list_tmp = [] # liste temporaire utilisée dans la boucle
    nb_movies = len(df_copy) # Nbre de films

    # Boucle sur les films de notre DataFrame ..._copy
    for i in range(nb_movies):
        list_tmp = df_copy["genres"][i].split(",")
            # Affectation la liste des genres de la ligne courant à la ligne temp
        df_copy["genres"][i] = list_tmp
            # Remplacement de la valeur dans la colonne "genres" par cette liste de genres
        # Vérification, pour chaque genre de cette liste, s'il est déjà présent dans le dictionnaire
        for str_genre in list_tmp:
            if not str_genre in dict_genres:
                # Si le genre n'est pas dans le dictionnaire, on l'ajoute
                dict_genres[str_genre] = 1
            else:
                # Si le genre est déjà dans le dictionnaire, on incrémente son nombre d'occurences
                dict_genres[str_genre] += 1
    
    # Création d'un DataFrame des genres
    df_genres = pd.DataFrame.from_dict(dict_genres, orient = 'index', columns = ['Occurences'])
    df_genres.sort_values(by = "Occurences", ascending = False, inplace = True)
    df_genres.reset_index(inplace = True)
    df_genres.rename(columns = {"index" : "Genre"}, inplace = True)
    df_genres["Selected"] = False
    
    # Changement de type de données de la colonne "Genre", passage en type "string"
    df_genres["Genre"] = df_genres["Genre"].astype("string")

    # Liste des genres que l'on veut supprimer
    list_genres_to_delete = ['Adult','News', 'Reality-TV', 'Talk-Show', 'Short', 'Game-Show']

    # Ajout d'une colonne "genres_to_delete" montrant s'il y a l'un des genres à supprimer dans la colonne "genres" (True)
    df_copy["genres_to_delete"] = df_copy.genres.apply(lambda x : any(gen in x for gen in list_genres_to_delete))
    
    # Suppression des lignes pour lesquelles "genres_to_delete" est True, c'est-à-dire
    # les lignes ayant un genre que l'on veut supprimer
    df_copy.drop(df_copy[df_copy.genres_to_delete == True].index, inplace = True)
    
    # Reset des index
    df_copy.reset_index(inplace = True, drop = True)
    
    # Suppression de la colonne temporaire "genres_to_delete"
    df_copy.drop("genres_to_delete", axis = 1, inplace = True)
    
    return df_copy, df_genres

@st.cache_data
def load_and_process_title_ratings():
    '''
    Charge et les données du fichier des "ratings"" lu sur le site d'IMDb, puis renvoie un
    DataFrame pandas contenant les données .

    Returns:
    -------
    pandas.DataFrame
        DataFrame contenant les données du fichier title.ratings.tsv.gz.

    Notes:
    ------
    La fonction charge le fichier title.ratings.tsv.gz à partir de l'URL spécifiée du site
    IMDb. Le DataFrame final contient toutes les données des notes et votes pour tous
    les titres.
    '''

    df_title_ratings = pd.read_csv(r"https://datasets.imdbws.com/title.ratings.tsv.gz",
                                   delimiter = '\t', low_memory = False)
    return df_title_ratings

@st.cache_data
def load_and_process_title_principals_and_name_basics():
    # Définition des "chunks"
    chunksize = 600000
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_principals = ['tconst', 'nconst', 'category']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.principals
    df_title_principals = pd.DataFrame()
    
    # Lecture du fichier csv title.principals.tsv avec des chunks
    #path_principals = r"https://datasets.imdbws.com/title.principals.tsv.gz"
    path_principals = r"C:/Données/d_ Wild Code School/d_ Projet 02/datasets/title_principals.tsv"
    df_chunks_principals = pd.read_csv(
        path_principals, usecols = columns_to_include_principals,
        dtype = {'tconst': 'string', 'nconst': 'string', 'category': 'category'}, delimiter = '\t',
        chunksize = chunksize)
    
    for chunk in df_chunks_principals:
        # Filtrer et traiter chaque morceau
        
        # Suppression des colonnes "ordering", "job", "characters"
        #chunk.drop(['ordering','job','characters'], axis = 1, inplace = True)
        
        # Suppression des lignes qui ne contiennent pas "actor" ou "actress" dans la colonne "categoy"
        chunk = chunk.drop(chunk[chunk['category'].isin(['actor','actress', 'director']) == False].index)
        
        # Suppression des doublons
        chunk.drop_duplicates(inplace = True)
        
        # Concaténer le morceau traité au DataFrame final
        df_title_principals = pd.concat([df_title_principals, chunk])
        
        # DataFrame avec uniquement les acteurs/actrisses
        df_title_principals_actors = df_title_principals[
            df_title_principals["category"].isin(["actor" , "actress"])]
        
        # DataFrame avec uniquement les réalisateurs
        df_title_principals_directors = df_title_principals[ df_title_principals["category"] == "director"]
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_name = ['nconst', 'primaryName']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.principals
    df_name_basics = pd.DataFrame()
    
    # Lecture du fichier csv name.basics.tsv avec des chunks
    #path_principals = r"https://datasets.imdbws.com/name.basics.tsv.gz"
    path_principals = r"C:/Données/d_ Wild Code School/d_ Projet 02/datasets/name_basics.tsv"
    df_chunks_name = pd.read_csv(
        path_principals, usecols = columns_to_include_name,
        dtype = {'nconst': 'string', 'primaryName': 'string'}, delimiter = '\t', chunksize = chunksize)
    
    for chunk in df_chunks_name:
        # Filtrer et traiter chaque morceau
        
        # Concaténer le morceau traité au DataFrame final
        df_name_basics = pd.concat([df_name_basics, chunk])
    
    # Fusion des deux DataFrame (actors)
    df_actors_movies = pd.merge(left = df_title_principals_actors, right = df_name_basics,
                                how = 'inner', left_on = "nconst", right_on = "nconst")
    
    # Fusion des deux DataFrame (directors)
    df_directors_movies = pd.merge(left = df_title_principals_directors, right = df_name_basics,
                                how = 'inner', left_on = "nconst", right_on = "nconst")
    
    # Fusion avec le DataFrame des title_ratings pour avoir les votes et notes des films des acteurs/trices
    df_actors_movies_ratings = pd.merge(left = df_actors_movies, right = df_title_ratings, how = "inner",
                                        left_on = "tconst", right_on = "tconst")
    
    # Fusion avec le DataFrame des title_ratings pour avoir les votes et notes des films des réalisateurs
    df_directors_movies_ratings = pd.merge(left = df_directors_movies, right = df_title_ratings, how = "inner",
                                        left_on = "tconst", right_on = "tconst")
    
    return df_actors_movies_ratings, df_directors_movies_ratings

@st.cache_data
def load_movies_fr_recent_years_from_github():
	with st.spinner('Import de du fichier movies_fr_recent_years.csv'):
		df_movie_fr_recent_years = pd.read_csv(
			"https://raw.githubusercontent.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films/main/movies_fr_recent_years.csv")
	return df_movie_fr_recent_years

@st.cache_data
def load_movies_fr_recent_years_trim_from_github():
	with st.spinner('Import de du fichier movies_fr_recent_years_trim.csv'):
		df_movie_fr_recent_years_trim = pd.read_csv(
			"https://raw.githubusercontent.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films/main/movies_fr_recent_years_trim.csv")
	return df_movie_fr_recent_years_trim

@st.cache_data
def load_genres_from_github():
	df_genres = pd.read_csv("https://raw.githubusercontent.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films/main/genres.csv")
	return df_genres

@st.cache_data
def load_movies_fr_from_1980_actors_from_github():
	with st.spinner('Import de du fichier movies_fr_from_1980_actors_ratings.csv'):
		df_movie_in_FR_from_1980_actor_rating = pd.read_csv(
			"https://raw.githubusercontent.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films/main/movies_fr_from_1980_actors_ratings.csv")
	return df_movie_in_FR_from_1980_actor_rating

@st.cache_data
def load_movies_fr_from_1980_directors_from_github():
	with st.spinner('Import de du fichier movies_fr_from_1980_directors_ratings.csv'):
		df_movies_Fr_from_1980_director_rating = pd.read_csv(
			"https://raw.githubusercontent.com/Miche5967/Projet_WCS_02_Systeme_recommandation_films/main/movies_fr_from_1980_directors_ratings.csv")
	return df_movies_Fr_from_1980_director_rating

# Top des x acteurs ayant le plus de votes, classés par note moyenne
@st.cache_data
def top_actors(nb_top_actors, sort_by_rating = False):
	# Définition d'un "top" des acteurs ayant le plus de votes
	df_actors_with_more_votes = df_group_actors_votes_ratings.sort_values(
		by = ['numVotes'], ascending = False).head(nb_top_actors)

	# Classement de ces acteurs ayant le plus de votes par le note moyenne pondérée
	if sort_by_rating:
		df_top_actors = df_actors_with_more_votes.sort_values(by = "weighted_rating", ascending = False).reset_index(drop = True)
	else:
		df_top_actors = df_actors_with_more_votes.reset_index(drop = True)

	return df_top_actors

# Top des x rélisateurs ayant le plus de vote classés par note moyenne
@st.cache_data
def top_directors(nb_top_directors, sort_by_rating = False):
	# Définition d'un "top" des réalisateurs ayant le plus de votes
	df_directors_with_more_votes = df_group_directors_votes_ratings.sort_values(
		by = ['numVotes'], ascending = False).head(nb_top_directors)

	# Classement de ces réalisateurs ayant le plus de votes par le note moyenne pondérée
	if sort_by_rating:
		df_top_directors = df_directors_with_more_votes.sort_values(
			by = "weighted_rating", ascending = False).reset_index(drop = True)
	else:
		df_top_directors = df_directors_with_more_votes.reset_index(drop = True)

	return df_top_directors

def keep_on_movie_analyse_page():
	st.session_state.radio = 'Analyses de films'

with st.spinner('Merci de patienter pendant le chargement des données. Cela peut prendre plusieurs minutes...'):
	if data_loading_type_from_github:
		df_movie_fr_recent_years = load_movies_fr_recent_years_from_github()
		df_movie_fr_recent_years_trim, df_genres = process_genres(df_movie_fr_recent_years)
		df_title_ratings = load_and_process_title_ratings()
		df_movie_in_FR_from_1980_actor_rating = load_movies_fr_from_1980_actors_from_github()
		df_movies_Fr_from_1980_director_rating = load_movies_fr_from_1980_directors_from_github()
	else:
		df_movie_fr_recent_years = load_and_process_title_akas_and_basics()
		df_movie_fr_recent_years_trim, df_genres = process_genres(df_movie_fr_recent_years)
		df_title_ratings = load_and_process_title_ratings()
		df_actors_movies_ratings, df_directors_movies_ratings = load_and_process_title_principals_and_name_basics()

if st.sidebar.radio('Choix de la page', ('Analyses de films', 'Recommandation de films'), key = "radio") == 'Analyses de films':
	st.header("Analyses de films")

	tab_genres, tab_actors, tab_directors = st.tabs(["Genres", "Actors/Actresses", "Directors"])

	with tab_genres:
		st.subheader("Analyse des genres")

		df_genres["Selected"] = False

		# Définition de 5 colonnes
		col_1, col_2, col_3, col_4, col_5 = st.columns(5)

		# Ajout des checkbox avec les noms des genres dans les différentes colonnes
		# La valeur de chaque checkbox est mise dans la colonne "Selected" de df_genres
		# Les premiers genres sont préselectionnés avec value = "True"
		with col_1:
			chk_Drama = st.checkbox("Drame", key = "chk_drama", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Drama", "Selected"] = chk_Drama
			chk_Comedy = st.checkbox("Comédie", key = "chk_comedy", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Comedy", "Selected"] = chk_Comedy
			chk_Action = st.checkbox("Action", key = "chk_action", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Action", "Selected"] = chk_Action
			chk_Thriller = st.checkbox("Thriller", key = "chk_thriller", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Thriller", "Selected"] = chk_Thriller
		with col_2:
			chk_Crime = st.checkbox("Policier", key = "chk_crime", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Crime", "Selected"] = chk_Crime
			chk_Romance = st.checkbox("Romance", key = "chk_romance", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Romance", "Selected"] = chk_Romance
			chk_Adventure = st.checkbox("Aventure", key = "chk_adventure", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Adventure", "Selected"] = chk_Adventure
			chk_Horror = st.checkbox("Horreur", key = "chk_horror", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Horror", "Selected"] = chk_Horror
		with col_3:
			chk_Mystery = st.checkbox("Mystery", key = "chk_mystery", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Mystery", "Selected"] = chk_Mystery
			chk_Fantasy = st.checkbox("Fantaisie", key = "chk_fantasy", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Fantasy", "Selected"] = chk_Fantasy
			chk_Biography = st.checkbox("Biographie", key = "chk_biography", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Biography", "Selected"] = chk_Biography
			chk_Family = st.checkbox("Famille", key = "chk_family", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Family", "Selected"] = chk_Family
		with col_4:
			chk_Sci_Fi = st.checkbox("S-F", key = "chk_sci_fi", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Sci-Fi", "Selected"] = chk_Sci_Fi
			chk_Animation = st.checkbox("Animation", key = "chk_animation", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Animation", "Selected"] = chk_Animation
			chk_History = st.checkbox("Historique", key = "chk_history", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "History", "Selected"] = chk_History
			chk_Music = st.checkbox("Musical", key = "chk_music", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Music", "Selected"] = chk_Music
		with col_5:
			chk_War = st.checkbox("Guerre", key = "chk_war", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "War", "Selected"] = chk_War
			chk_Sport = st.checkbox("Sport", key = "chk_sport", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Sport", "Selected"] = chk_Sport
			chk_Western = st.checkbox("Western", key = "chk_western", on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Western", "Selected"] = chk_Western

		# Création d'un DataFrame avec la colonne "genres" éclatée
		df_movie_title_fr_recent_years_exploded = df_movie_fr_recent_years_trim.explode(column = "genres")
		df_movie_title_fr_recent_years_exploded["genres"] = df_movie_title_fr_recent_years_exploded["genres"].astype("string")

		# Création d'un DataFrame des genres ne contenant que les genres à conserver
		df_genres_trim = df_genres.loc[df_genres.Selected]

		# Création d'un DataFrame avec uniquement les genres à conserver définis plus haut
		df_movie_title_fr_recent_years_exploded_trim = df_movie_title_fr_recent_years_exploded.loc[
			df_movie_title_fr_recent_years_exploded["genres"].isin(df_genres_trim["Genre"])]

		# Création d'un nouveau DataFrame comme résultat de cette jointure
		# Merge de type inner join (how = 'inner') sur les champs "tconst" de chaque table
		# pour récupérer les lignes pour lesquelles les notes et nombes de votes sont disponibles.
		df_movie_title_fr_recent_years_exploded_ratings = pd.merge(
		    left = df_movie_title_fr_recent_years_exploded_trim, right = df_title_ratings, how = 'inner',
		    left_on = "tconst", right_on = "tconst")

		# Groupement des données par genre et année et aggréagation
		# Création d'un nouveau DataFrame pour ne conserver que les colonnes qui nous intéressent
		df_years_genres_ratings = df_movie_title_fr_recent_years_exploded_ratings[
			["startYear", "genres", "runtimeMinutes", "averageRating", "numVotes"]]

		# Nouvelle colonne "weighted_rating" pour le calcul des moyennes pondérées
		df_years_genres_ratings["weighted_rating"] = df_years_genres_ratings.averageRating * df_years_genres_ratings.numVotes

		# Nouvelle colonne "nbMovies" pour le calcul du nombre de films par catégorie
		df_years_genres_ratings["nbMovies"] = 1

		# Création d'un nouveau DataFrame df_group_years_genres en groupant les données du précédent par année et par genre
		# avec comme aggrégation les somme des nombres de votes et des valeurs de la nouvelle colonne weighted_rating"
		df_group_years_genres = df_years_genres_ratings.groupby(
		    by = ["startYear", "genres"]).agg(
		    {"numVotes" : "sum", "weighted_rating" : "sum", "nbMovies": "sum", "runtimeMinutes" : "mean"})

		# Division de la colonne "weighted_rating"  par le nombre de vote pour calculer la moyenne pondérée
		df_group_years_genres.weighted_rating = df_group_years_genres.weighted_rating / df_group_years_genres.numVotes

		# Création d'un DataFrame pour le tracé, en ré-initialisant les index
		df_group_years_genres_to_plot = df_group_years_genres.reset_index()

		
		### Tracés ###

		# Dictionnaire des couleurs par genre
		dict_genres_colors_map = {'Drama' : 'plum', 'Comedy' : 'dodgerblue', 'Documentary' : 'green', 'Action' : 'gold',
			'Thriller' : 'darkred', 'Crime' : 'red', 'Romance' : 'deeppink', 'Adventure' : 'lightgreen', 'Horror' : 'indigo',
			'Mystery' : 'lightgrey', 'Biography' : 'lavender', 'Fantasy' : 'lightskyblue', 'Family' : 'yellow', 'Sci-Fi' : 'silver',
			'Animation' : 'salmon', 'History' : 'slategrey', 'Music' : 'darkorchid', 'War' : 'darkgreen', 'Adult' : 'pink',
			'Sport' : 'lawngreen', 'Musical' : 'orchid', 'Western' : 'navajowhite'}

		# Line plot de la moyenne pondérée (y) par an (x) et par genre (catégorie)
		st.markdown("### Courbe de la moyenne pondérée (y) par an (x) et par genre (catégorie)")
		fig_2 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "weighted_rating",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			width = 1000, height = 600, line_shape ='spline', color_discrete_map = dict_genres_colors_map,
			labels = {"startYear": "Année", "weighted_rating": "Moyenne pondérée", "genres": "Genre"},
			title = "Note moyenne pondérée des films par an et par genre")

		fig_2.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white', yaxis=dict(range=[4, max(df_group_years_genres_to_plot['weighted_rating'])]))
		fig_2.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range=[1979, 2023])
		fig_2.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_2, use_container_width = False)

		# Line plot du nombre de votes (y) par an (x) et par genre (catégorie)
		st.markdown("### Courbe de la moyenne pondérée (y) par an (x) et par genre (catégorie)")
		fig_1 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "numVotes",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			width = 1000, height = 600, line_shape ='spline', color_discrete_map = dict_genres_colors_map,
			labels = {"startYear": "Année", "numVotes": "Nombre de votes", "genres": "Genre"},
			title = "Nombre de votes moyen des films par an et par genre")

		fig_1.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_1.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range=[1979, 2023])
		fig_1.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_1, use_container_width = False)

		# Evolution du nombre de films sortis par an et par genre
		# Line plot du nombre de films produits par an et par genre
		st.markdown("### Courbe du nombre de films produits par an et par genre")
		fig_3 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "nbMovies",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			labels = {"startYear": "Année", "nbMovies": "Nombre de films", "genres": "Genre"},
			title = "Nombre de films produits par an et par genre", width = 1000, height = 600,
			color_discrete_map = dict_genres_colors_map)

		fig_3.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_3.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range = [1979, 2023])
		fig_3.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range = [0, 1100])

		# Affichage dans Streamlit
		st.plotly_chart(fig_3, use_container_width = False)

		# Line plot de la durée moyenne des films par an et par genre
		st.markdown("### Courbe de la durée moyenne des films par an et par genre")
		fig_4 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "runtimeMinutes",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			labels = {"startYear": "Année", "nbMovies": "Nombre de films", "genres": "Genre"},
			title = "Durée moyenne des films par an et par genre", width = 1000, height = 600,
			color_discrete_map = dict_genres_colors_map)

		fig_4.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_4.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range = [1979, 2023])
		fig_4.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_4, use_container_width = False)

		#df_sample = df_group_years_genres_to_plot.sample(20)
		#st.dataframe(df_sample)
	

	with tab_actors:
		st.subheader("Acteurs et Actrices")

		# Fusion du DataFrame des notes des films des acteurs avec le DataFrame des films

		# Création d'un nouveau DataFrame par fusion des DataFrame des notes des films des acteurs
		# avec le DataFrame des films,
		# SI les données n'ont pas été directement chargées depuis github
		if not data_loading_type_from_github:
			df_movie_in_FR_from_1980_actor_rating = pd.merge(left = df_movie_fr_recent_years_trim,
				right =df_actors_movies_ratings, how = "inner", left_on = "tconst", right_on = "tconst")

			# Ajout d'une colonne "weighted_rating" pour le calcul de la moyenne pondérée des notes des films
			df_movie_in_FR_from_1980_actor_rating['weighted_rating'] = \
				df_movie_in_FR_from_1980_actor_rating['averageRating'] * df_movie_in_FR_from_1980_actor_rating['numVotes']
			
			# Ajout d'une colonne "nb_movies" pour le calcul du nombre de films par acteur
			df_movie_in_FR_from_1980_actor_rating['nb_movies'] = 1

		# Moyenne pondérée des notes des films des acteurs

		# Création d'un nouveau DataFrame df_actors_votes_ratings en conservant les colonnes qui nous intéressent
		df_actors_votes_ratings = df_movie_in_FR_from_1980_actor_rating[
			["nconst", "primaryName", "numVotes", "weighted_rating", "nb_movies", "startYear"]]

		# Groupement des données par acteur, en sommant les autres colonnes
		df_group_actors_votes_ratings = df_actors_votes_ratings.groupby(by = ["primaryName"]).agg(
			{"numVotes" : "sum", "weighted_rating" : "sum", "nb_movies" : "sum"})
		
		# Calcul de la moyenne des notes pondérée en divisant weighted_rating par le nombre de votes
		df_group_actors_votes_ratings["weighted_rating"] = \
			df_group_actors_votes_ratings["weighted_rating"] / df_group_actors_votes_ratings["numVotes"]
		
		# Reset des index pour remettre le nom de l'acteur/actrice en colonne
		df_group_actors_votes_ratings.reset_index(inplace = True)

		#df_top_15_actors = top_actors(15)


		### Tracés ###

		# Bar chart des acteurs ayant le plus de votes, classés par note moyenne
		nb_actors = 20
		fig_5 = px.bar(top_actors(nb_actors), x = 'primaryName', y = 'weighted_rating', height = 600, width = 1000,
			title = f'{nb_actors} acteurs ayant le plus de votes classés par note moyenne',
			labels = {"primaryName": "Nom", "weighted_rating": "Note moyenne pondérée", "numVotes": "Nombre de votes"},
			color_discrete_sequence = ['lightblue'], hover_data = ['numVotes', 'nb_movies'])

		fig_5.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_5.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2)
		fig_5.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_5, use_container_width = False)

		
		st.markdown("### Top 200 des acteurs dans les films ayant le plus de votes")

		# Définition d'un top 200 des acteurs ayant participé aux films qui ont le plus de votes
		df_top_200_actors = top_actors(200)

		# Affichage des acteurs du top 200
		st.dataframe(df_top_200_actors)

	with tab_directors:
		st.subheader("Réalisateurs")

		# Fusion du DataFrame des notes des films des acteurs avec le DataFrame des films

		# Création d'un nouveau DataFrame par fusion des DataFrame des notes des films des réalisateurs
		# avec le DataFrame des films
		# SI les données n'ont pas été directement chargées depuis github
		if not data_loading_type_from_github:
			df_movies_Fr_from_1980_director_rating = pd.merge(left = df_movie_fr_recent_years_trim,
				right = df_directors_movies_ratings, how = "inner", left_on = "tconst", right_on = "tconst")

			# Ajout d'une colonne "weighted_rating" pour le calcul de la moyenne pondérée des notes des films
			df_movies_Fr_from_1980_director_rating['weighted_rating'] = \
				df_movies_Fr_from_1980_director_rating['averageRating'] * df_movies_Fr_from_1980_director_rating['numVotes']

			# Ajout d'une colonne "nb_movies" pour le calcul du nombre de films par acteur
			df_movies_Fr_from_1980_director_rating['nb_movies'] = 1

		# Moyenne pondérée des notes des films des réalisateurs
		# Création d'un nouveau DataFrame df_directors_votes_ratings en conservant les colonnes qui nous intéressent
		df_directors_votes_ratings = df_movies_Fr_from_1980_director_rating[
			["nconst", "primaryName", "numVotes", "weighted_rating", "nb_movies", "startYear"]]

		# Groupement des données par réalisateur, en sommant les autres colonnes
		df_group_directors_votes_ratings = df_directors_votes_ratings.groupby(by = ["primaryName"]).agg(
			{"numVotes" : "sum", "weighted_rating" : "sum", "nb_movies" : "sum"})

		# Calcul de la moyenne des notes pondérée en divisant weighted_rating par le nombre de votes
		df_group_directors_votes_ratings["weighted_rating"] = \
			df_group_directors_votes_ratings["weighted_rating"] / df_group_directors_votes_ratings["numVotes"]

		# Reset des index pour remettre le nom de l'acteur/actrice en colonne
		df_group_directors_votes_ratings.reset_index(inplace = True)

		#df_top_15_directors = top_directors(15)

		### Tracés ###

		# Bar chart des réalisateurs ayant le plus de votes, classés par note moyenne
		nb_directors = 20
		fig_7 = px.bar(top_directors(nb_directors), x = 'primaryName', y = 'weighted_rating', height = 600, width = 1000,
			title = f'{nb_directors} réalisateurs ayant le plus de votes classés par note moyenne',
			labels = {"primaryName": "Nom", "weighted_rating": "Note moyenne pondérée", "numVotes": "Nombre de votes"},
			color_discrete_sequence = ['lightblue'], hover_data = ['numVotes', 'nb_movies'])

		fig_7.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_7.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2)
		fig_7.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_7, use_container_width = False)

		
		st.markdown("### Top 50 des réalisateurs les films ayant le plus de votes")

		# Définition d'un top 50 des réalisateurs ayant réalisé les films qui ont le plus de votes
		df_top_50_directors = top_directors(50)

		# Affichage des réalisateur du top 50
		st.dataframe(df_top_50_directors)

else:
    st.header("Recommandations de films") 

    # Ajout du magnifique GIF des minions 
    #gif = "https://media.giphy.com/media/11sBLVxNs7v6WA/giphy.gif"
    #st.image(gif, use_column_width=True, width = 300)

    
    ### Ajout d'une nouvelle colonne "recommandé" ###

    # Création d'un DataFrame temporaire avec uniquement l'id du film et le(s) noms du(des) réalisateur(s)
    # Et changement de nom de la colonne "primaryName"
    @st.cache_data
    def get_movies_directors():
    	df_movies_directors = df_movies_Fr_from_1980_director_rating[["tconst", "primaryName"]].rename(
            columns = {"primaryName" : "dir_primaryName"})
    	return df_movies_directors

    # Création d'un DataFrame temporaire avec uniquement l'id du film et le(s) noms du(des) acteur/actrice(s)
    # Et changement de nom de la colonne "primaryName"
    #@st.cache_data
    def get_movies_actors():
        df_movies_actors = df_movie_in_FR_from_1980_actor_rating[["tconst", "primaryName"]].rename(
            columns = {"primaryName" : "act_primaryName"})
        return df_movies_actors

    # DataFrame avec uniquement l'id du film et le(s) noms du(des) réalisateur(s)
    df_movies_directors = get_movies_directors()
    # DataFrame avec uniquement l'id du film et le(s) noms du(des) acteur/actrice(s)
    df_movies_actors = get_movies_actors()

    # Nouveau DataFrame créé en fusionnant df_movie_in_FR_from_1980_actor_rating avec df_movies_directors 
    df_movie_fr_from_1980_ratings = pd.merge(
        left = df_movie_in_FR_from_1980_actor_rating, right = df_movies_directors,
        how = "inner", left_on = "tconst", right_on = "tconst").rename(columns = {"primaryName": "act_primaryName"})

    # Ajout d'une nouvelle colonne "recommandé" avec valeur 0 par défaut
    df_movie_fr_from_1980_ratings["recommended"] = 0

    ## Critères pour passer "recommandé" à 1 :
    # 60 min < durée < 180 min ET
        # 1. nbre votes >= 100 K ET note moyenne >= 7 OU
        # 2. nbre votes >= 10 K ET note moyenne >= 5 ET acteur dans le top 200 des acteurs les plus populaires
        # 3. nbre votes >= 10 K ET note moyenne >= 5 ET réalisateur dans le top 50 des réalisateurs les plus populaires

    df_top_200_actors = top_actors(200)
    df_top_50_directors = top_directors(50)

    # 1ère condition pour passer "recommandé" à 1
    df_movie_fr_from_1980_ratings.loc[
        (df_movie_fr_from_1980_ratings['runtimeMinutes'] >= 60) & (df_movie_fr_from_1980_ratings['runtimeMinutes'] <= 180) &
        (df_movie_fr_from_1980_ratings['numVotes'] >= 100000) & (df_movie_fr_from_1980_ratings['averageRating'] >= 7),
        'recommended'] = 1

    # 2e condition pour passer "recommandé" à 1
    df_movie_fr_from_1980_ratings.loc[
        (df_movie_fr_from_1980_ratings['runtimeMinutes'] >= 60) & (df_movie_fr_from_1980_ratings['runtimeMinutes'] <= 180) &
        (df_movie_fr_from_1980_ratings['numVotes'] >= 10000) & (df_movie_fr_from_1980_ratings['averageRating'] >= 5) &
        (df_movie_fr_from_1980_ratings['act_primaryName'].isin(df_top_200_actors['primaryName'])),
        'recommended'] = 1

    # 3e condition pour passer "recommandé" à 1
    df_movie_fr_from_1980_ratings.loc[
        (df_movie_fr_from_1980_ratings['runtimeMinutes'] >= 60) & (df_movie_fr_from_1980_ratings['runtimeMinutes'] <= 180) &
        (df_movie_fr_from_1980_ratings['numVotes'] >= 10000) & (df_movie_fr_from_1980_ratings['averageRating'] >= 5) &
        (df_movie_fr_from_1980_ratings['dir_primaryName'].isin(df_top_50_directors['primaryName'])),
        'recommended'] = 1

    # On ne conserve que les colonnes qui nous intéressent, on supprime notamment les infos des acteurs/actrices
    df_movie_fr_from_1980_ratings = df_movie_fr_from_1980_ratings[
        ["tconst", "startYear", "runtimeMinutes", "genres", "title", "averageRating", "numVotes", "recommended"]]

    # Création d'un DataFrame en groupant les films et leurs caractéristiques,
    # et en sommant la colonne "recommended"
    df_movie_fr_from_1980_ratings_recommendation = df_movie_fr_from_1980_ratings.groupby(
        by = ["tconst", "startYear", "runtimeMinutes", "genres", "title",  "averageRating", "numVotes"],
        as_index = False)["recommended"].sum()

    # Changement de type de la colonne "recommended" en integer
    df_movie_fr_from_1980_ratings_recommendation["recommended"] = \
        df_movie_fr_from_1980_ratings_recommendation["recommended"].astype(int)

    # Changement de type de la colonne "startYear" en integer
    df_movie_fr_from_1980_ratings_recommendation["startYear"] = \
        df_movie_fr_from_1980_ratings_recommendation["startYear"].astype(int)

    # Changement de type de la colonne "runtimeMinutes" en integer
    df_movie_fr_from_1980_ratings_recommendation["runtimeMinutes"] = \
        df_movie_fr_from_1980_ratings_recommendation["runtimeMinutes"].astype(int)

    # Changement de type de la colonne "title" en string
    df_movie_fr_from_1980_ratings_recommendation["title"] = \
        df_movie_fr_from_1980_ratings_recommendation["title"].astype("string")

    # Fonction pour supprimer les "[" "]" à une chaîne de caractères
    def remove_brackets(string):
        if len(string) > 0:
            if string[0] == "[":
                string = string[1:]
            if string[-1] == "]":
                string = string[:-1]
            return string
        else:
            return ""
    
    # Suppression des crochets ("brackets" "[" et "]") de la chaîne de caractères des genres
    df_movie_fr_from_1980_ratings_recommendation["genres"] = \
        df_movie_fr_from_1980_ratings_recommendation["genres"].apply(remove_brackets)

    # Reset des index
    df_movie_fr_from_1980_ratings_recommendation.reset_index(drop = True, inplace = True)



    ### Machine Learning ###

    # Recommandation de films

    # Saisie d'un film par l'utilisateur
    titre_film = st.text_input("Veuillez renseigner un titre")
    #titre_film = titre_film

    #df_algo["lower_title"] = df_algo["title"].str.lower()

    # Choix d'un film
    #titre_film = "Casino"

    #
    if len(titre_film) > 0:
    	# Création du DataFrame du film choisi df_chosen_movie
    	df_chosen_movie = df_movie_fr_from_1980_ratings_recommendation[
    		df_movie_fr_from_1980_ratings_recommendation["title"] == titre_film]
    	#print(df_movie_fr_from_1980_ratings_recommendation)
    	#print(df_chosen_movie)

    	# Définition des genres du film choisi
    	genres_film = df_movie_fr_from_1980_ratings_recommendation.loc[
    		df_movie_fr_from_1980_ratings_recommendation["title"] == titre_film, "genres"].values[0]

    	# Définition de la liste des genres du film choisi
    	list_genres_film = []
    	for genre in df_genres.Genre:
    		if genre in genres_film:
    			list_genres_film.append(genre)

    	# Fonction vérifiant si un genre de la liste est compris dans une chaîne
    	def check_genre_match(string):
    		nb_matches = 0
    		for genre in list_genres_film:
    			if genre in string:
    				nb_matches += 1
    		return nb_matches

    	# Copie du DataFrame de base dans un df temporaire df_movie_tmp
    	df_movie_tmp = df_movie_fr_from_1980_ratings_recommendation.copy()

    	# Suppression du film choisi par l'utilisateur du DataFrame d'entraînement
    	df_movie_tmp = df_movie_tmp[df_movie_tmp["title"] != titre_film]

    	# Application de la fonction pour vérifier si les genres du film choisi "matchent"
    	# avec les genres des films du DataFrame
    	df_movie_tmp["nb_genre_matches"] = df_movie_tmp["genres"].apply(check_genre_match)

    	# On ne garde que les films pour lesquels le genre "matche" avec le film choisi
    	nb_mini_genre_matches = 50
    	if len(df_movie_tmp[df_movie_tmp["nb_genre_matches"] == len(list_genres_film)]) >= nb_mini_genre_matches:
    		df_movie_tmp = df_movie_tmp[df_movie_tmp["nb_genre_matches"] == len(list_genres_film)]
    	elif len(list_genres_film) > 1:
    		df_movie_tmp = df_movie_tmp[df_movie_tmp["nb_genre_matches"] >= len(list_genres_film) - 1]  
    	else:
    		df_movie_tmp = df_movie_tmp[df_movie_tmp["nb_genre_matches"] == len(list_genres_film)]

    	# Définition des variable explicatives X, ici nos variables quantitatives (numériques)
    	X = df_movie_tmp[["startYear", "runtimeMinutes", "averageRating", "numVotes", "recommended"]]

    	# Création et entraînement du modèle "scaler" pour standardiser nos variables
    	#scaler = StandardScaler().fit(X)

    	# Transformation des données par le modèle scaler
    	#X_scaled = scaler.transform(X)

    	# Création et entraînement du modèle "NearestNeighbors" (plus proches voisins)
    	# Choix du nombre de voisins
    	k = 50
    	modelNN = NearestNeighbors(n_neighbors = k).fit(X)
    	#modelNN = NearestNeighbors(n_neighbors = k).fit(X_scaled)

    	# kneighbors renvoie les distances et indices des k plus proches voisins renoyés par le modèle :
    	neighbors = modelNN.kneighbors(
    		df_movie_fr_from_1980_ratings_recommendation.loc[
    			df_movie_fr_from_1980_ratings_recommendation['title'] == titre_film, X.columns])

    	# Index des plus proches voisins
    	arr_closest_movies_index = neighbors[1][0]

    	# DataFrame des films recommandés
    	df_recommended_movies = df_movie_tmp.iloc[arr_closest_movies_index]
    	df_recommended_movies = df_recommended_movies[
    		["startYear", "runtimeMinutes", "genres", "title", "averageRating", "numVotes", "recommended"]]
    	if len(df_recommended_movies[df_recommended_movies.recommended > 0]) > 10:
    		df_recommended_movies = df_recommended_movies[df_recommended_movies.recommended > 0]
    	df_recommended_movies.rename(
    		columns = {"startYear" : "Année", "runtimeMinutes" : "Durée", "genres" : "Genres", "title": "Titre",
    		"averageRating" : "Note moy.", "numVotes" : "Nbre de votes", "recommended" : "Recommandé"},
    		inplace = True)
    	if len(df_recommended_movies) > 10:
    		df_recommended_movies = df_recommended_movies.head(10)

    	st.dataframe(df_recommended_movies)



#st.sidebar.markdown('<div style="height: 350px;"></div>', unsafe_allow_html=True)
#image = Image.open("https://github.com/Miche5967/movie_analyse_recommendation_streamlit_app/blob/master/analystes_redim.png?raw=true")
#container = st.sidebar.container()
#st.sidebar.image(image, use_column_width=True)

image_url2 = "https://github.com/Miche5967/movie_analyse_recommendation_streamlit_app/blob/master/analystes_redim.png?raw=true"
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.write(" ", unsafe_allow_html=True)
st.sidebar.image(image_url2, use_column_width=True)