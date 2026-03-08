## Mini Project 1 - Part 1: Getting Familiar with Word Embeddings.
# This assignment introduces students to text similarity measures using cosine similarity and sentence embeddings. 
# Students will implement and compare different methods for computing and analyzing text similarity using GloVe, Sentence Transformers, and OpenAI Embeddings.

#Learning Objectives
#By the end of this assignment, students will:
#Understand how cosine similarity is used to measure text similarity.
#Learn to encode sentences using GloVe embeddings, Sentence Transformers, and OpenAI embeddings.
#Compare the performance of different embedding techniques (small vs large models).
#Create a Web interface for your model
#Analyze intuitive examples like "Chocolate Milk" vs "Milk Chocolate" to understand embedding behavior

# Context: In this part, you are going to play around with some commonly used pretrained text embeddings for text search. For example, GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Pretrained on 
# 2 billion tweets with vocabulary size of 1.2 million. Download from [Stanford NLP](http://nlp.stanford.edu/data/glove.twitter.27B.zip). 
# Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.

### Import necessary libraries: here you will use streamlit library to run a text search demo, please make sure to install it.
import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle
import os
import gdown
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import math
from openai import OpenAI  # Added for OpenAI embeddings


### Some predefined utility functions for you to load the text embeddings

# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path="Data/embeddings.pkl"):
    with open(glove_path, "rb") as f:
        embeddings_dict = pickle.load(f, encoding="latin1")

    return embeddings_dict


def get_model_id_gdrive(model_type):
    if model_type == "25d":
        word_index_id = "13qMXs3-oB9C6kfSRMwbAtzda9xuAUtt8"
        embeddings_id = "1-RXcfBvWyE-Av3ZHLcyJVsps0RYRRr_2"
    elif model_type == "50d":
        embeddings_id = "1DBaVpJsitQ1qxtUvV1Kz7ThDc3az16kZ"
        word_index_id = "1rB4ksHyHZ9skes-fJHMa2Z8J1Qa7awQ9"
    elif model_type == "100d":
        word_index_id = "1-oWV0LqG3fmrozRZ7WB1jzeTJHRUI3mq"
        embeddings_id = "1SRHfX130_6Znz7zbdfqboKosz-PfNvNp"
        
    return word_index_id, embeddings_id


def download_glove_embeddings_gdrive(model_type):
    # Get glove embeddings from google drive
    word_index_id, embeddings_id = get_model_id_gdrive(model_type)

    # Use gdown to get files from google drive
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"

    # Download word_index pickle file
    print("Downloading word index dictionary....\n")
    gdown.download(id=word_index_id, output=word_index_temp, quiet=False)

    # Download embeddings numpy file
    print("Donwloading embedings...\n\n")
    gdown.download(id=embeddings_id, output=embeddings_temp, quiet=False)


# @st.cache_data()
def load_glove_embeddings_gdrive(model_type):
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"

    # Load word index dictionary
    word_index_dict = pickle.load(open(word_index_temp, "rb"), encoding="latin")

    # Load embeddings numpy
    embeddings = np.load(embeddings_temp)

    return word_index_dict, embeddings


@st.cache_resource()
def load_sentence_transformer_model(model_name):
    sentenceTransformer = SentenceTransformer(model_name)
    return sentenceTransformer


@st.cache_resource()
def load_openai_client():
    """
    Load OpenAI client (cached to avoid multiple initializations)
    Make sure to set OPENAI_API_KEY environment variable
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    return OpenAI(api_key=api_key)


def get_openai_embeddings(sentence, model_name="text-embedding-3-small"):
    """
    Get OpenAI embeddings for a sentence
    
    Available models:
    - text-embedding-3-small: 1536 dimensions, faster and cheaper
    - text-embedding-3-large: 3072 dimensions, higher performance
    
    Args:
        sentence: Input text to embed
        model_name: OpenAI embedding model to use
    
    Returns:
        numpy array of embeddings
    """
    client = load_openai_client()
    if client is None:
        # Return zero vector if API key not available
        if model_name == "text-embedding-3-small":
            return np.zeros(1536)
        else:  # text-embedding-3-large
            return np.zeros(3072)
    
    try:
        response = client.embeddings.create(
            input=sentence,
            model=model_name
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error getting OpenAI embeddings: {e}")
        if model_name == "text-embedding-3-small":
            return np.zeros(1536)
        else:
            return np.zeros(3072)


def get_sentence_transformer_embeddings(sentence, model_name="all-MiniLM-L6-v2"):
    """
    Get sentence transformer embeddings for a sentence
    """
    # 384 dimensional embedding
    # Default model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  

    sentenceTransformer = load_sentence_transformer_model(model_name)

    try:
        return sentenceTransformer.encode(sentence)
    except:
        if model_name == "all-MiniLM-L6-v2":
            return np.zeros(384)
        else:
            return np.zeros(512)


def get_glove_embeddings(word, word_index_dict, embeddings, model_type):
    """
    Get glove embedding for a single word
    """
    if word.lower() in word_index_dict:
        return embeddings[word_index_dict[word.lower()]]
    else:
        return np.zeros(int(model_type.split("d")[0]))


def get_category_embeddings(embeddings_metadata):
    """
    Get embeddings for each category
    1. Split categories into words
    2. Get embeddings for each word
    """
    model_name = embeddings_metadata["model_name"]
    embedding_model = embeddings_metadata["embedding_model"]
    
    cache_key = "cat_embed_" + embedding_model + "_" + model_name
    st.session_state[cache_key] = {}
    
    for category in st.session_state.categories.split(" "):
        if embedding_model == "openai":
            if category not in st.session_state[cache_key]:
                st.session_state[cache_key][category] = get_openai_embeddings(category, model_name=model_name)
        elif model_name:
            if category not in st.session_state[cache_key]:
                st.session_state[cache_key][category] = get_sentence_transformer_embeddings(category, model_name=model_name)
        else:
            if category not in st.session_state[cache_key]:
                st.session_state[cache_key][category] = get_sentence_transformer_embeddings(category)


def update_category_embeddings(embeddings_metadata):
    """
    Update embeddings for each category
    """
    get_category_embeddings(embeddings_metadata)




### Plotting utility functions
    
def plot_piechart(sorted_cosine_scores_items):
    sorted_cosine_scores = np.array([
            sorted_cosine_scores_items[index][1]
            for index in range(len(sorted_cosine_scores_items))
        ]
    )
    categories = st.session_state.categories.split(" ")
    categories_sorted = [
        categories[sorted_cosine_scores_items[index][0]]
        for index in range(len(sorted_cosine_scores_items))
    ]
    fig, ax = plt.subplots()
    ax.pie(sorted_cosine_scores, labels=categories_sorted, autopct="%1.1f%%")
    st.pyplot(fig)  # Figure


def plot_piechart_helper(sorted_cosine_scores_items):
    sorted_cosine_scores = np.array(
        [
            sorted_cosine_scores_items[index][1]
            for index in range(len(sorted_cosine_scores_items))
        ]
    )
    categories = st.session_state.categories.split(" ")
    categories_sorted = [
        categories[sorted_cosine_scores_items[index][0]]
        for index in range(len(sorted_cosine_scores_items))
    ]
    fig, ax = plt.subplots(figsize=(3, 3))
    my_explode = np.zeros(len(categories_sorted))
    my_explode[0] = 0.2
    if len(categories_sorted) == 3:
        my_explode[1] = 0.1  # explode this by 0.2
    elif len(categories_sorted) > 3:
        my_explode[2] = 0.05
    ax.pie(
        sorted_cosine_scores,
        labels=categories_sorted,
        autopct="%1.1f%%",
        explode=my_explode,
    )

    return fig


def plot_piecharts(sorted_cosine_scores_models):
    scores_list = []
    categories = st.session_state.categories.split(" ")
    index = 0
    for model in sorted_cosine_scores_models:
        scores_list.append(sorted_cosine_scores_models[model])
        # scores_list[index] = np.array([scores_list[index][ind2][1] for ind2 in range(len(scores_list[index]))])
        index += 1

    if len(sorted_cosine_scores_models) == 2:
        fig, (ax1, ax2) = plt.subplots(2)

        categories_sorted = [
            categories[scores_list[0][index][0]] for index in range(len(scores_list[0]))
        ]
        sorted_scores = np.array(
            [scores_list[0][index][1] for index in range(len(scores_list[0]))]
        )
        ax1.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

        categories_sorted = [
            categories[scores_list[1][index][0]] for index in range(len(scores_list[1]))
        ]
        sorted_scores = np.array(
            [scores_list[1][index][1] for index in range(len(scores_list[1]))]
        )
        ax2.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

    st.pyplot(fig)


def plot_alatirchart(sorted_cosine_scores_models):
    models = list(sorted_cosine_scores_models.keys())
    tabs = st.tabs(models)
    figs = {}
    for model in models:
        figs[model] = plot_piechart_helper(sorted_cosine_scores_models[model])

    for index in range(len(tabs)):
        with tabs[index]:
            st.pyplot(figs[models[index]])


### Your Part To Complete: Follow the instructions in each function below to complete the similarity calculation between text embeddings

# Task I: Compute Cosine Similarity
def cosine_similarity(x, y):
    """
    Exponentiated cosine similarity
    1. Compute cosine similarity
    2. Exponentiate cosine similarity
    3. Return exponentiated cosine similarity
    (20 pts)
    """
    ##################################
    ### TODO: Add code here (10 pts) ###
    ##################################
    
    # x and y should be 1D numpy arrays
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        # Avoid division by zero; treat similarity as zero if any vector is all zeros
        return 0.0
    cosine_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # Exponentiate cosine similarity
    exp_cosine_sim = np.exp(cosine_sim)
    return exp_cosine_sim
    
    
# Task II: Average Glove Embedding Calculation
def averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type="50d"):
    """
    Get averaged glove embeddings for a sentence
    1. Split sentence into words
    2. Get embeddings for each word
    3. Add embeddings for each word
    4. Divide by number of words
    5. Return averaged embeddings
    (30 pts)
    """
    embedding_dim = int(model_type.split("d")[0])
    embedding = np.zeros(embedding_dim)
    
    ##################################
    ##### TODO: Add code here (20 pts) #####
    ##################################
    # Split sentence into words (make all lowercase for consistency)
    words = sentence.lower().split()
    count = 0

    # Get embeddings for each word
    for word in words:
        # Find index for word in dictionary
        idx = word_index_dict.get(word, None)
        if idx is not None:
            # Add embedding for each word
            embedding += embeddings[idx]
            count += 1

    # Divide by number of found words (avoid division by zero)
    if count > 0:
        embedding /= count
    else:
        # If none of the words were found, return zero vector (already zeros)
        pass

    # Return averaged embeddings
    return embedding


# Task III: Sort the cosine similarity
def get_sorted_cosine_similarity(embeddings_metadata):
    """
    Get sorted cosine similarity between input sentence and categories
    
    Steps:
    1. Get embeddings for input sentence
    2. Get embeddings for categories (if not found, update category embeddings)
    3. Compute cosine similarity between input sentence and categories
    4. Sort cosine similarity
    5. Return sorted cosine similarity
    
    (50 pts)

    NOTE: This function must handle THREE different embedding types:
    - GloVe Embeddings (word averaging)
    - OpenAI Embeddings (API-based, small & large models)
    - Sentence Transformer Embeddings (context-aware)
    
    IMPORTANT: 
    - Each embedding type requires different methods to get embeddings
    - Use caching (st.session_state) to avoid redundant computations
    - Return results sorted by similarity (highest first)
    
    Args:
        embeddings_metadata: Dictionary containing:
            - "embedding_model": Type of model ("glove", "openai", or "transformers")
            - For GloVe: "word_index_dict", "embeddings", "model_type"
            - For OpenAI/Transformers: "model_name"
    
    Returns:
        List of tuples: [(category_index, similarity_score), ...]
        Sorted in descending order by similarity_score
    
    Available helper functions:
        - averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type)
        - get_openai_embeddings(sentence, model_name)
        - get_sentence_transformer_embeddings(sentence, model_name)
        - get_category_embeddings(embeddings_metadata)
        - cosine_similarity(x, y)
    """
    # Get input sentence from session state
    sentence = st.session_state.text_search
    categories = st.session_state.categories.split(" ")
    cosine_sim = {}
    
    if embeddings_metadata["embedding_model"] == "glove":
        # Extract GloVe-specific parameters
        word_index_dict = embeddings_metadata["word_index_dict"]
        embeddings = embeddings_metadata["embeddings"]
        model_type = embeddings_metadata["model_type"]

        ##########################################
        ## TODO: Implement GloVe similarity calculation (15 pts)
        ##########################################
        # define cache key
        cache_key = "cat_embed_glove_" + model_type

        # Manually handle category embeddings for GloVe (don't call get_category_embeddings)
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}
        for category in categories:
            if category not in st.session_state[cache_key]:
                st.session_state[cache_key][category] = averaged_glove_embeddings_gdrive(
                    category, word_index_dict, embeddings, model_type
                )

        # Get embeddings for input sentence
        sentence_embedding = averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type)

    elif embeddings_metadata["embedding_model"] == "openai":
        # Extract OpenAI-specific parameters
        model_name = embeddings_metadata["model_name"]
        cache_key = "cat_embed_openai_" + model_name
        
        ##########################################
        ## TODO: Implement OpenAI similarity calculation (15 pts)
        ##########################################
        # Get embeddings for input sentence
        sentence_embedding = get_openai_embeddings(sentence, model_name)
        
    else:  # Sentence transformers
        # Extract Sentence Transformer-specific parameters
        model_name = embeddings_metadata["model_name"]
        cache_key = "cat_embed_transformers_" + (model_name if model_name else "default")
        
        ##########################################
        ## TODO: Implement Sentence Transformer similarity calculation (15 pts)
        ##########################################
        # Get embeddings for input sentence
        sentence_embedding = get_sentence_transformer_embeddings(sentence, model_name)
        

    ##########################################
    ## TODO: Sort and return results (5 pts)
    ##########################################
    # Get category embeddings (only for non-GloVe models)
    if embeddings_metadata["embedding_model"] != "glove":
        get_category_embeddings(embeddings_metadata)


    # Compute cosine similarity between input sentence and categories
    if sentence_embedding is not None:  # Protect against None values
        # Check if sentence_embedding contains NaN or inf
        if np.isnan(sentence_embedding).any() or np.isinf(sentence_embedding).any():
            # If sentence embedding is invalid, set all similarities to 0
            for idx in range(len(categories)):
                cosine_sim[idx] = 0.0
        else:
            for idx, category in enumerate(categories):
                # Get embedding for category
                cat_embed = st.session_state[cache_key][category]
                # Check if embeddings are valid (not None and not containing NaN)
                if cat_embed is not None and not np.isnan(cat_embed).any() and not np.isinf(cat_embed).any():
                    sim_score = cosine_similarity(sentence_embedding, cat_embed)
                    # Replace NaN or inf with 0
                    if np.isnan(sim_score) or np.isinf(sim_score):
                        cosine_sim[idx] = 0.0
                    else:
                        cosine_sim[idx] = sim_score
                else:
                    cosine_sim[idx] = 0.0
    else:
        # If sentence_embedding is None, set all similarities to 0
        for idx in range(len(categories)):
            cosine_sim[idx] = 0.0

    # Sort cosine similarity
    sorted_cosine_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
    # Return sorted cosine similarity
    return sorted_cosine_sim


### Below is the main function, creating the app demo for text search engine using the text embeddings.

if __name__ == "__main__":
    ### Text Search ###
    ### There will be Bonus marks of 10% for the teams that submit a URL for your deployed web app. 
    ### Bonus: You can also submit a publicly accessible link to the deployed web app.

    st.sidebar.title("Embedding Models")
    st.sidebar.markdown(
        """
    Compare different embedding models:
    - **GloVe**: Pretrained on 2 billion tweets
    - **Sentence Transformers**: Fine-tuned on sentence similarity tasks
    - **OpenAI Small**: text-embedding-3-small (1536d)
    - **OpenAI Large**: text-embedding-3-large (3072d)
    
    [GloVe Paper](http://nlp.stanford.edu/data/glove.twitter.27B.zip) | 
    [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
    """
    )

    model_type = st.sidebar.selectbox("Choose the model", ("25d", "50d", "100d"), index=1)


    st.title("Search Based Retrieval Demo")
    st.subheader(
        "Pass in space separated categories you want this search demo to be about."
    )
    # st.selectbox(label="Pick the categories you want this search demo to be about...",
    # options=("Flowers Colors Cars Weather Food", "Chocolate Milk", "Anger Joy Sad Frustration Worry Happiness", "Positive Negative"),
    # key="categories"
    # )
    st.text_input(
        label="Categories", key="categories", value="Flowers Colors Cars Weather Food"
    )
    print(st.session_state["categories"])
    print(type(st.session_state["categories"]))
    # print("Categories = ", categories)
    # st.session_state.categories = categories

    st.subheader("Pass in an input word or even a sentence")
    text_search = st.text_input(
        label="Input your sentence",
        key="text_search",
        value="Roses are red, trucks are blue, and Seattle is grey right now",
    )
    sentence = text_search
    # st.session_state.text_search = text_search

    # Download glove embeddings if it doesn't exist
    embeddings_path = "embeddings_" + str(model_type) + "_temp.npy"
    word_index_dict_path = "word_index_dict_" + str(model_type) + "_temp.pkl"
    if not os.path.isfile(embeddings_path) or not os.path.isfile(word_index_dict_path):
        print("Model type = ", model_type)
        glove_path = "Data/glove_" + str(model_type) + ".pkl"
        print("glove_path = ", glove_path)

        # Download embeddings from google drive
        with st.spinner("Downloading glove embeddings..."):
            download_glove_embeddings_gdrive(model_type)


    # Load glove embeddings
    word_index_dict, embeddings = load_glove_embeddings_gdrive(model_type)


    # Find closest word to an input word
    if st.session_state.text_search:
        results_dict = {}
        
        # Glove embeddings
        print("Glove Embedding")
        embeddings_metadata = {
            "embedding_model": "glove",
            "word_index_dict": word_index_dict,
            "embeddings": embeddings,
            "model_type": model_type,
        }
        with st.spinner("Obtaining Cosine similarity for Glove..."):
            sorted_cosine_sim_glove = get_sorted_cosine_similarity(embeddings_metadata)
            results_dict["glove_" + str(model_type)] = sorted_cosine_sim_glove

        # Sentence transformer embeddings
        print("Sentence Transformer Embedding")
        embeddings_metadata = {
            "embedding_model": "transformers", 
            "model_name": "all-MiniLM-L6-v2"
        }
        with st.spinner("Obtaining Cosine similarity for 384d sentence transformer..."):
            sorted_cosine_sim_transformer = get_sorted_cosine_similarity(embeddings_metadata)
            results_dict["sentence_transformer_384"] = sorted_cosine_sim_transformer

        # OpenAI Small embeddings
        print("OpenAI Small Embedding")
        embeddings_metadata = {
            "embedding_model": "openai", 
            "model_name": "text-embedding-3-small"
        }
        with st.spinner("Obtaining Cosine similarity for OpenAI Small (1536d)..."):
            sorted_cosine_sim_openai_small = get_sorted_cosine_similarity(embeddings_metadata)
            results_dict["openai_small_1536"] = sorted_cosine_sim_openai_small

        # OpenAI Large embeddings
        print("OpenAI Large Embedding")
        embeddings_metadata = {
            "embedding_model": "openai", 
            "model_name": "text-embedding-3-large"
        }
        with st.spinner("Obtaining Cosine similarity for OpenAI Large (3072d)..."):
            sorted_cosine_sim_openai_large = get_sorted_cosine_similarity(embeddings_metadata)
            results_dict["openai_large_3072"] = sorted_cosine_sim_openai_large

        # Results and Plot Pie Chart for all models
        print("Categories are: ", st.session_state.categories)
        st.subheader(
            "Closest category between: "
            + st.session_state.categories
            + " as per different Embeddings"
        )

        # Display results in tabs
        plot_alatirchart(results_dict)
        
        # Add comparison table
        st.markdown("---")
        st.subheader("Detailed Comparison")
        
        categories = st.session_state.categories.split(" ")
        comparison_data = []
        
        for model_name, scores in results_dict.items():
            top_category_idx = scores[0][0]
            top_category = categories[top_category_idx]
            top_score = scores[0][1]
            comparison_data.append({
                "Model": model_name,
                "Top Category": top_category,
                "Confidence Score": f"{top_score:.4f}"
            })
        
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.table(df)

        st.write("")
        st.write(
            "Demo developed by Bo-Han Chang Chien (https://www.linkedin.com/in/bohancc/)"
        )
        

        
