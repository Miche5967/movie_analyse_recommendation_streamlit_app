# Imports des librairies
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time

# Changement de la largeur de la page pour l'élargir
st.set_page_config(layout="wide")

# Titre de la page
st.title('Movie analyse and recommendation')

# Affichage (temporaire) du "session_state"
"st.session.state (*juste pour voir, à supprimer ensuite*) :", st.session_state

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

    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_akas = ['titleId', 'title', 'region']
    
    # Lecture du fichier title.akas.tsv.gz
    df_title_akas_pyarrow_trim_dtype = pd.read_csv(
        r"https://datasets.imdbws.com/title.akas.tsv.gz", usecols = columns_to_include_akas,
        dtype = {'titleId': 'string', 'title': 'string', 'region': 'string'}, delimiter = '\t', engine = "pyarrow")
    
    # Filtre sur la colonne "region" pour ne garder que la valeur "FR"
    df_title_akas_fr = df_title_akas_pyarrow_trim_dtype.loc[df_title_akas_pyarrow_trim_dtype["region"] == "FR"]
    
    # Suppression de la colonne "region"
    df_title_akas_fr_trim = df_title_akas_fr.drop(columns = "region")
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.basics
    columns_to_include_basics = ['tconst', 'titleType', 'startYear', 'runtimeMinutes', 'genres']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.basics
    df_title_basics_recent_years = pd.DataFrame()
    
    # Définition des "chunks" pour optimiser la lecture
    chunksize = 500000
    # Lecture du fichier title.basics.tsv.gz par morceau
    df_chunks = pd.read_csv(
        r"https://datasets.imdbws.com/title.basics.tsv.gz", usecols = columns_to_include_basics,
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

def keep_on_movie_analyse_page():
	st.session_state.radio = 'Analyses de films'

with st.spinner('Merci de patienter pendant le chargement des données. Cela peut prendre 2 à 5 minutes...'):
	df_movie_fr_recent_years = load_and_process_title_akas_and_basics()
	df_movie_fr_recent_years_trim, df_genres = process_genres(df_movie_fr_recent_years)
	df_title_ratings = load_and_process_title_ratings()

#if st.sidebar.button('Movie analyses', key = "btn_analyse"):
if st.sidebar.radio('Choix de la page', ('Analyses de films', 'Recommandation de films'), key = "radio") == 'Analyses de films':
	st.header("Movie analyses")

	tab_genres, tab_actors, tab_directors = st.tabs(["Genres", "Actors/Actresses", "Directors"])

	with tab_genres:
		st.subheader("Genres analyses")

		df_genres["Selected"] = False

		# Définition de 5 colonnes
		col_1, col_2, col_3, col_4, col_5 = st.columns(5)

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
			chk_Romance = st.checkbox("Romance", key = "chk_romance", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Romance", "Selected"] = chk_Romance
			chk_Adventure = st.checkbox("Aventure", key = "chk_adventure", value = True, on_change = keep_on_movie_analyse_page)
			df_genres.loc[df_genres["Genre"] == "Adventure", "Selected"] = chk_Adventure
			chk_Horror = st.checkbox("Horreur", key = "chk_horror", value = True, on_change = keep_on_movie_analyse_page)
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

		#st.dataframe(df_genres.head(10))
		
		# Création d'un DataFrame avec la colonne "genres" éclatée
		df_movie_title_fr_recent_years_exploded = df_movie_fr_recent_years_trim.explode(column = "genres")
		df_movie_title_fr_recent_years_exploded["genres"] = df_movie_title_fr_recent_years_exploded["genres"].astype("string")

		# Définition d'une liste de genres à conserver
		list_genres_to_keep = ['Drama', 'Comedy', 'Action', 'Thriller', 'Crime',
			'Romance', 'Adventure', 'Horror', 'Mystery', 'Fantasy']
		
		# Création d'un DataFrame des genres ne contenant que les genres à conserver
		#df_genres_trim = df_genres.loc[df_genres.Genre.isin(list_genres_to_keep)]
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
		df_years_genres_ratings = df_movie_title_fr_recent_years_exploded_ratings[["startYear", "genres", "averageRating", "numVotes"]]

		# Nouvelle colonne "weighted_rating" pour le calcul des moyennes pondérées
		df_years_genres_ratings["weighted_rating"] = df_years_genres_ratings.averageRating * df_years_genres_ratings.numVotes

		# Création d'un nouveau DataFrame df_group_years_genres en groupant les données du précédent par année et par genre
		# avec comme aggrégation les somme des nombres de votes et des valeurs de la nouvelle colonne weighted_rating"
		df_group_years_genres = df_years_genres_ratings.groupby(
		    by = ["startYear", "genres"]).agg({"numVotes" : "sum", "weighted_rating" : "sum"})

		# Division de la colonne "weighted_rating"  par le nombre de vote pour calculer la moyenne pondérée
		df_group_years_genres.weighted_rating = df_group_years_genres.weighted_rating / df_group_years_genres.numVotes

		# Création d'un DataFrame pour le tracé, en ré-initialisant les index
		df_group_years_genres_to_plot = df_group_years_genres.reset_index()

		# Tracé
		fig = px.scatter(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "weighted_rating",
		                 size = "numVotes", color = "genres", color_discrete_sequence = px.colors.qualitative.Light24,
		                 width = 1000,
		                 labels = {"startYear": "Year", "weighted_rating": "Weighted rating", "genres": "Movie genre",
		                          "numVotes": "Number of votes"},
		                 title = "Ratings and number of votes of movies produced per year and genre", height = 700)

		fig.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
		                           'font' : dict(size = 24)},
		                 plot_bgcolor = 'white')

		fig.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
		                 gridcolor = 'lightgrey', griddash = 'dash',
		                 range=[1979, 2023])
		fig.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
		                 gridcolor = 'lightgrey', griddash = 'dash')

		st.plotly_chart(fig, use_container_width = False)

		df_sample = df_group_years_genres_to_plot.sample(20)
		st.dataframe(df_sample)
	
	with tab_actors:
		st.subheader("Actresses & actors")

	with tab_directors:
		st.subheader("Directors")

else:
	st.header("Movie recommendation")
	st.write("A venir...")