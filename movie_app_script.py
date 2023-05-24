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
#"st.session.state (*juste pour voir, à supprimer ensuite*) :", st.session_state

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
    df_chunks_akas = pd.read_csv(
    	r"https://datasets.imdbws.com/title.akas.tsv.gz", usecols = columns_to_include_akas,
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
    # Définition des "chunks" pour optimiser la lecture
    chunksize = 600000
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_principals = ['tconst', 'nconst', 'category']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.principals
    df_title_principals = pd.DataFrame()
    
    # Lecture du fichier csv title.akas.tsv avec des chunks
    df_chunks_principals = pd.read_csv(
        r"https://datasets.imdbws.com/title.principals.tsv.gz", usecols = columns_to_include_principals,
        dtype = {'tconst': 'string', 'nconst': 'string', 'category': 'category'}, delimiter = '\t',
        chunksize = chunksize)
    
    for chunk in df_chunks_principals:
        # Filtrer et traiter chaque morceau
        
        # Suppression des colonnes "ordering", "job", "characters"
        #chunk.drop(['ordering','job','characters'], axis = 1, inplace = True)
        
        # Suppression des lignes qui ne contiennent pas "actor" ou "actress" dans la colonne "categoy"
        chunk = chunk.drop(chunk[chunk['category'].isin(['actor','actress']) == False].index)
        
        # Suppression des doublons
        chunk.drop_duplicates(inplace = True)
        
        # Concaténer le morceau traité au DataFrame final
        df_title_principals = pd.concat([df_title_principals, chunk])
    
    # Définition des colonnes à conserver lors de la lecture du fichier csv title.akas
    columns_to_include_name = ['nconst', 'primaryName']
    
    # Initialisation d'un DataFrame pour la lecture du fichier title.principals
    df_name_basics = pd.DataFrame()
    
    # Lecture du fichier csv title.akas.tsv avec des chunks
    df_chunks_name = pd.read_csv(
        r"https://datasets.imdbws.com/name.basics.tsv.gz", usecols = columns_to_include_name,
        dtype = {'nconst': 'string', 'primaryName': 'string'},delimiter = '\t', chunksize = chunksize)
    
    for chunk in df_chunks_name:
        # Filtrer et traiter chaque morceau
        
        # Concaténer le morceau traité au DataFrame final
        df_name_basics = pd.concat([df_name_basics, chunk])
    
    # Fusion des deux DataFrame
    df_actors_movies = pd.merge(left = df_title_principals, right = df_name_basics,
                                how = 'inner', left_on = "nconst", right_on = "nconst")
    
    # Fusion avec le DataFrame des title_ratings pour avoir les votes et notes des films des acteurs/trices
    df_actors_movies_ratings = pd.merge(left = df_actors_movies, right = df_title_ratings, how = "inner",
                                        left_on = "tconst", right_on = "tconst")
    
    return df_actors_movies_ratings

def keep_on_movie_analyse_page():
	st.session_state.radio = 'Analyses de films'

with st.spinner('Merci de patienter pendant le chargement des données. Cela peut prendre plusieurs minutes...'):
	df_movie_fr_recent_years = load_and_process_title_akas_and_basics()
	df_movie_fr_recent_years_trim, df_genres = process_genres(df_movie_fr_recent_years)
	df_title_ratings = load_and_process_title_ratings()
	df_actors_movies_ratings = load_and_process_title_principals_and_name_basics()

#if st.sidebar.button('Movie analyses', key = "btn_analyse"):
if st.sidebar.radio('Choix de la page', ('Analyses de films', 'Recommandation de films'), key = "radio") == 'Analyses de films':
	st.header("Movie analyses")

	tab_genres, tab_actors, tab_directors = st.tabs(["Genres", "Actors/Actresses", "Directors"])

	with tab_genres:
		st.subheader("Genres analyses")

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

		# Tracés

		# Dictionnaire des couleurs par genre
		dict_genres_colors_map = {'Drama' : 'plum', 'Comedy' : 'dodgerblue', 'Documentary' : 'green', 'Action' : 'gold',
			'Thriller' : 'darkred', 'Crime' : 'red', 'Romance' : 'deeppink', 'Adventure' : 'lightgreen', 'Horror' : 'indigo',
			'Mystery' : 'lightgrey', 'Biography' : 'lavender', 'Fantasy' : 'lightskyblue', 'Family' : 'yellow', 'Sci-Fi' : 'silver',
			'Animation' : 'salmon', 'History' : 'slategrey', 'Music' : 'darkorchid', 'War' : 'darkgreen', 'Adult' : 'pink',
			'Sport' : 'lawngreen', 'Musical' : 'orchid', 'Western' : 'navajowhite'}

		# Bubble plot de la moyenne pondérée (y) par an (x) et par genre (catégorie),
		# avec le nombre de votes pour la taille des bulles
		st.markdown("### Bubble plot de la moyenne pondérée (y) par an (x) et par genre (catégorie)")
		fig_1 = px.scatter(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "weighted_rating",
		                 size = "numVotes", color = "genres", color_discrete_sequence = px.colors.qualitative.Light24,
		                 width = 1000, height = 600, color_discrete_map = dict_genres_colors_map,
		                 labels = {"startYear": "Year", "weighted_rating": "Weighted rating", "genres": "Movie genre",
		                          "numVotes": "Number of votes"},
		                 title = "Ratings and number of votes of movies produced per year and genre")

		fig_1.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
		                           'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_1.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
		                 gridcolor = 'lightgrey', griddash = 'dash', range=[1979, 2023])
		fig_1.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
		                 gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_1, use_container_width = False)

		# Line plot de la moyenne pondérée (y) par an (x) et par genre (catégorie)
		st.markdown("### Courbe de la moyenne pondérée (y) par an (x) et par genre (catégorie)")
		fig_2 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "weighted_rating",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			width = 1000, line_shape='spline', color_discrete_map = dict_genres_colors_map,
			labels = {"startYear": "Year", "weighted_rating": "Weighted rating", "genres": "Movie genre"},
			title = "Ratings of movies produced per year and genre")

		fig_2.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_2.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range=[1979, 2023])
		fig_2.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_2, use_container_width = False)

		
		# Evolution du nombre de films sortis par an et par genre
		# Line plot du nombre de films produits par an et par genre
		st.markdown("### Courbe du nombre de films produits par an et par genre")
		fig_3 = px.line(data_frame = df_group_years_genres_to_plot, x = "startYear", y = "nbMovies",
			color = "genres", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			labels = {"startYear": "Year", "nbMovies": "Number of movies", "genres": "Movie genres"},
			title = "Number of movies produced per year and genre", width = 1000, height = 600,
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
			labels = {"startYear": "Year", "nbMovies": "Number of movies", "genres": "Movie genres"},
			title = "Average duration of movies per year and genre", width = 1000, height = 600,
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
		st.subheader("Actresses & actors")

		# Fusion du DataFrame des notes des films des acteurs avec le DataFrame des films

		# Création d'un nouveau DataFrame par fusion des DataFrame des notes des films des acteurs
		# avec le DataFrame des films
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

		# Top des x acteurs ayant le plus de votes, classés par note moyenne

		def top_actors(nb_top_actors):
			# Définition d'un "top" des acteurs ayant le plus de votes
			df_actors_with_more_votes = df_group_actors_votes_ratings.sort_values(
				by = ['numVotes'], ascending = False).head(nb_top_actors)

			# Classement de ces acteurs ayant le plus de votes par le note moyenne pondérée
			df_top_actors = df_actors_with_more_votes.sort_values(by = "weighted_rating", ascending = False).reset_index(drop = True)

			return df_top_actors

		df_top_15_actors = top_actors(15)


		# Tracés

		# Bar chart des acteurs ayant le plus de votes, classés par note moyenne
		fig_5 = px.bar(df_top_15_actors, x = 'primaryName', y = 'weighted_rating',
			title = 'Les 15 acteurs les mieux notés avec le plus de votes',
			color_discrete_sequence = ['lightblue'], hover_data = ['numVotes'])

		fig_5.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_5.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2)
		fig_5.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_5, use_container_width = False)

		# Evolution des notes moyennes des acteurs ayant le plus de votes

		# Groupement des données par acteur et par an, en sommant les autres colonnes
		df_group_actors_years_votes_ratings = df_actors_votes_ratings.groupby(by = ["primaryName", "startYear"]).agg(
			{"numVotes" : "sum", "weighted_rating" : "sum", "nb_movies" : "sum"})

		# Calcul de la moyenne des notes pondérée en divisant weighted_rating par le nombre de votes
		df_group_actors_years_votes_ratings["weighted_rating"] = \
			df_group_actors_years_votes_ratings["weighted_rating"] / df_group_actors_years_votes_ratings["numVotes"]

		# Reset des index pour remettre le nom de l'acteur/actrice en colonne
		df_group_actors_years_votes_ratings.reset_index(inplace = True)

		# Création nouveau DataFrame en ne conservant que les "top" acteurs
		df_top_15_actors_evolution = df_group_actors_years_votes_ratings[
			df_group_actors_years_votes_ratings["primaryName"].isin(df_top_15_actors["primaryName"])]

		# Tracé
		fig_6 = px.line(data_frame = df_top_15_actors_evolution, x = "startYear", y = "weighted_rating",
			color = "primaryName", color_discrete_sequence = px.colors.qualitative.Light24, markers = True,
			labels = {"weighted_rating" : "Note moyenne pondérée", "startYear": "Année",
				"nb_movies": "Nombre de films", "primaryName": "Nom"},
			title = "Evolution de la note moyenne pondérée des meilleurs acteurs",
			width = 1000, height = 600, hover_data = ["numVotes"])

		fig_6.update_layout(title = {'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
			'font' : dict(size = 24)}, plot_bgcolor = 'white')
		fig_6.update_xaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash', range = [1979, 2023])
		fig_6.update_yaxes(ticks = 'outside', showline = True, linecolor = 'black', linewidth = 2,
			gridcolor = 'lightgrey', griddash = 'dash')

		# Affichage dans Streamlit
		st.plotly_chart(fig_6, use_container_width = False)

	with tab_directors:
		st.subheader("Directors")

else:
	st.header("Movie recommendation")
	st.write("A venir...")