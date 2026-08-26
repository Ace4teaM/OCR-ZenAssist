import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import pickle
import os
from pathlib import Path

database_path = Path("database", os.environ["EMBEDDING_MODEL"])
index_filename = str(Path(str(database_path), os.environ["FAISS_INDEX_FILENAME"]).resolve())
meta_filename = str(Path(str(database_path), os.environ["FAISS_META_FILENAME"]).resolve())

database_path.parent.mkdir(parents=True, exist_ok=True)

print("index_filename", index_filename)
print("meta_filename", meta_filename)
#
# vectorise les documents convertis (Sentence-BERT)
#


@lru_cache(maxsize=1)
def get_model():
    print("Chargement du modèle...", flush=True)
    return SentenceTransformer(os.environ["EMBEDDING_MODEL"])


def make_embeddings(text: str | list[str]) -> np.ndarray:
    """
    Vectorise le texte en utilisant le modèle Sentence-BERT.

    Retourne l'embedding de type <numpy.ndarray>
    """
    model = get_model() # en cache pour éviter de recharger le modèle à chaque appel

    # Générer les embeddings
    return model.encode(text, convert_to_numpy=True)

def load_database(
) -> pd.DataFrame:
    # charge la base de données et les metadatas
    with open(meta_filename, "rb") as f:
        metadata = pickle.load(f)

    index = faiss.read_index(index_filename)
    return metadata, index

def save_database(
    metadata,
    index
) -> pd.DataFrame:
    # Sauvegarde de l'index sur le disque
    faiss.write_index(index, index_filename)

    with open(meta_filename, "wb") as f:
        pickle.dump(metadata, f)

def make_database(
    train_data: pd.DataFrame
) -> pd.DataFrame:
    # ----------------------------------------------------------------
    # Création des embeddings
    # ----------------------------------------------------------------

    claims = train_data["Consumer Claim"].tolist()

    embeddings = make_embeddings(claims)

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalisation pour utiliser la similarité cosinus
    faiss.normalize_L2(embeddings)

    # ----------------------------------------------------------------
    # Création de l'index
    # ----------------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"Nombre de vecteurs : {index.ntotal}")

    metadata = train_data[
        [
            "Consumer Claim",
            "Tag"
        ]
    ].reset_index(drop=True)

    return metadata, index

def retrieve_examples(
    metadata, 
    index,
    claim: str,
    k: int = 5
) -> pd.DataFrame:
    """
    Recherche les réclamations les plus similaires dans l'index FAISS.

    Args:
        claim (str):
            Réclamation à utiliser comme requête.

        k (int, optional):
            Nombre d'exemples similaires à récupérer.

    Returns:
        pd.DataFrame:
            DataFrame contenant les réclamations similaires,
            leur catégorie et leur score de similarité.
    """

    # Création de l'embedding de la requête
    query_embedding = make_embeddings([claim])

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalisation
    faiss.normalize_L2(query_embedding)

    # Recherche des K voisins les plus proches
    scores, indices = index.search(
        query_embedding,
        k
    )

    # Récupération des métadonnées
    examples = metadata.iloc[
        indices[0]
    ].copy()

    examples["Similarity"] = scores[0]

    return examples