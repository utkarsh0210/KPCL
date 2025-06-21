import pandas as pd
import numpy as np

#filepath = r"C:\Users\utkun\Desktop\KPCL\ItemMaster.csv"

# Assuming 'Description' is the column containing text data


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Create a TF-IDF vectorizer
# vectorizer = TfidfVectorizer(stop_words='english')

# # Fit and transform the descriptions
# tfidf_matrix = vectorizer.fit_transform(descriptions)

# def find_best_matches(description_input, descriptions, tfidf_matrix, vectorizer):
#     """
#     Finds the top n best matches for a given description input.
#     Args:
#         description_input (str): The input description to match.
#         descriptions (list): A list of all descriptions in the dataset.
#         tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix of the descriptions.
#         vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
#         n (int): The number of best matches to return.

#     Returns:
#         list: A list of tuples containing the best matching descriptions and their similarity scores.
#     """
#     # Transform the input description
#     input_tfidf = vectorizer.transform([description_input])
#     # Calculate cosine similarity between the input and all descriptions
#     cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
#     # Get the indices of the top n most similar descriptions
#     top_n_indices = cosine_similarities.argsort()[:][::-1]
#     # Get the top n descriptions and their similarity scores
#     best_matches = [(descriptions[i],item_codes[i], long_desc[i], cosine_similarities[i]) for i in top_n_indices]
#     return best_matches

# def start_search(text_input, limit = None, df):

#     descriptions = df['Description'].dropna().tolist()
#     item_codes = df['Item number'].dropna().tolist()
#     long_desc = df['Long Description'].tolist()
#     # Create an input text field for the description
#     description_input = text_input #@param {type:"string"}
#     scores = []
#     codes = []
#     item_desc = []
#     desc_long = []
#     filtered_dict = []
#     if description_input and description_input != "Enter description here":
#         # Find the best matches
#         start_time = time.time()
#         # Find the best matches using TF-IDF cosine similarity
#         best_matches = find_best_matches(description_input, descriptions, tfidf_matrix, vectorizer)
#         end_time = time.time()
#         retrieval_time = end_time - start_time
#         threshold_Score = 0.4    
#         count = sum(1 for _, _,_, score in best_matches if score > threshold_Score)
#         # Print the best matches with a similarity threshold
#         #print(f"\nBest matches for '{description_input}' (retrieval time: {retrieval_time:.4f} seconds):")
#         for match,code, l_d, score in best_matches:
#             if score > threshold_Score:
#                 scores.append(score)
#                 codes.append(code)
#                 item_desc.append(match)
#                 desc_long.append(l_d)
#                 #print(f"-Code: {code} Description: {match} (Similarity: {score:.4f})")
#             else:
#                 continue
#         #print(f"Total records: {count}")
#     else:
#         print("\nPlease enter a description to find matches.")
#     filtered_dict = list(zip(codes, item_desc, desc_long, scores))
#     if limit is not None or limit < count:
#         filtered_dict = filtered_dict[:limit]
#     return filtered_dict, retrieval_time, count


def start_search(text_input, df, limit = None):

    descriptions = df['Description'].dropna().tolist()
    item_codes = df['Item number'].dropna().tolist()
    long_desc = df['Long Description'].tolist()

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the descriptions
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    def find_best_matches(description_input, descriptions, tfidf_matrix, vectorizer):
        """
        Finds the top n best matches for a given description input.
        Args:
            description_input (str): The input description to match.
            descriptions (list): A list of all descriptions in the dataset.
            tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix of the descriptions.
            vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
            n (int): The number of best matches to return.

        Returns:
            list: A list of tuples containing the best matching descriptions and their similarity scores.
        """
        # Transform the input description
        input_tfidf = vectorizer.transform([description_input])
        # Calculate cosine similarity between the input and all descriptions
        cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
        # Get the indices of the top n most similar descriptions
        top_n_indices = cosine_similarities.argsort()[:][::-1]
        # Get the top n descriptions and their similarity scores
        best_matches = [(descriptions[i],item_codes[i], long_desc[i], cosine_similarities[i]) for i in top_n_indices]
        return best_matches
    
    if not text_input:
        return [], 0.0, 0

    scores, codes, item_desc, desc_long = [], [], [], []
    if text_input and text_input != "Enter description here":
        start_time = time.time()
        # Find the best matches using TF-IDF cosine similarity
        best_matches = find_best_matches(text_input, descriptions, tfidf_matrix, vectorizer)
        end_time = time.time()
        retrieval_time = end_time - start_time
        threshold_Score = 0.4

        count = sum(1 for _, _,_, score in best_matches if score > threshold_Score)
        # Print the best matches with a similarity threshold
        #print(f"\nBest matches for '{description_input}' (retrieval time: {retrieval_time:.4f} seconds):")
        for match,code, l_d, score in best_matches:
            if score > threshold_Score:
                scores.append(score)
                codes.append(code)
                item_desc.append(match)
                desc_long.append(l_d)
                #print(f"-Code: {code} Description: {match} (Similarity: {score:.4f})")
            else:
                continue
        #print(f"Total records: {count}")
    else:
        print("\nPlease enter a description to find matches.")
    filtered_dict = list(zip(codes, item_desc, desc_long, scores))
    if limit is not None or limit < count:
        filtered_dict = filtered_dict[:limit]
    return filtered_dict, retrieval_time, count