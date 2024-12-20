from typing import List
import os

import numpy as np
import pandas as pd
import py7zr
from bs4 import BeautifulSoup
from evaluate import load
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
from umap import UMAP


def extract_data(data_dir: str):
    with py7zr.SevenZipFile(os.path.join(data_dir, "cooking.stackexchange.com.7z"), mode='r') as archive:
        archive.extractall(path=data_dir)


def load_data(data_dir: str):
    data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".xml"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "lxml")
                attributes = [row.attrs for row in soup.find_all("row")]
                data[filename[:-4]] = pd.DataFrame(attributes)
    return data


def clean_html(html_content):
    if not isinstance(html_content, str):
        return None
    soup = BeautifulSoup(html_content, "lxml")
    text_content = soup.get_text()
    return text_content


def calculate_bertscore(
    df: pd.DataFrame,
    prediction_column: str,
    reference_column: str,
    metric: str = "f1",
    batch_size: int = 50
):
    bertscore = load("bertscore")
    num_batches = len(df) // batch_size + (len(df) % batch_size > 0)
    output = []

    for i in tqdm(
        range(num_batches),
        desc=f"Calculating BERTScore (pred: {prediction_column}; ref: {reference_column}; metric: {metric})"
    ):
        start = i * batch_size
        end = start + batch_size

        batch_predictions = df[prediction_column].iloc[start:end].tolist()
        batch_references = df[reference_column].iloc[start:end].tolist()

        results = bertscore.compute(
            predictions=batch_predictions,
            references=batch_references,
            lang="en",
            device="cuda"
        )
        output.extend(results[metric])

    return output


class Embedder:
    def __init__(self, model: str = "BAAI/bge-m3", reduce_dim: int = None):
        self.model = SentenceTransformer(model)
        self.reduce_dim = reduce_dim or self.model.get_sentence_embedding_dimension()
        self.reducer = UMAP(n_components=self.reduce_dim)

    def embed(self, text: str | List[str], show_progress: bool = False):
        if isinstance(text, str):
            embeddings = self.model.encode(text)
        else:
            embeddings = self.model.encode(text, show_progress_bar=show_progress)
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        return reduced_embeddings


def preprocess_data(
    data: dict,
    n_negative_samples_per_link: int = 1,
    add_tags: bool = False,
    add_post_text: bool = False,
    include_pooled_comments: bool = False,
    add_bertscore: bool = False,
    add_title_embedding: bool = False,
    title_embedding_dim: int = None,
    add_body_embedding: bool = False,
    body_embedding_dim: int = None,
):
    post_links = data["PostLinks"]
    posts = data["Posts"][data["Posts"]["posttypeid"] == "1"]
    users = data["Users"]
    votes = data["Votes"]
    tags = data["Tags"]
    comments = data["Comments"]

    # Feature Engineering
    # 1. Extract features from Posts
    posts_features = ['id', 'score', 'viewcount', 'answercount', 'commentcount']
    if add_post_text:
        posts_features.extend(["body", "title"])
    posts_features = posts[posts_features]
    if add_post_text:
        posts_features['body'] = posts_features['body'].map(clean_html)
    posts_features = posts_features.rename(columns={'id': 'postid'})

    if add_title_embedding:
        title_embedder = Embedder(reduce_dim=title_embedding_dim)
        title_embeddings = title_embedder.embed(posts_features['title'].tolist(), show_progress=True)
        for i in range(title_embeddings.shape[1]):
            posts_features[f'title_embedding_{i}'] = [embedding[i] for embedding in title_embeddings]
    if add_body_embedding:
        body_embedder = Embedder(reduce_dim=body_embedding_dim)
        body_embeddings = body_embedder.embed(posts_features['body'].tolist(), show_progress=True)
        for i in range(body_embeddings.shape[1]):
            posts_features[f'body_embedding_{i}'] = [embedding[i] for embedding in body_embeddings]

    # 2. Extract features from Users
    users_features = users[['id', 'reputation', 'views', 'upvotes', 'downvotes']]
    users_features = users_features.rename(columns={'id': 'userid'})

    # 3. Aggregate Votes
    votes_agg = votes.groupby('postid').agg({'votetypeid': 'count'}).reset_index()
    votes_agg = votes_agg.rename(columns={'votetypeid': 'vote_count'})

    # 4. Aggregate Comments
    comments_agg = (
        comments.astype({"score": "int32", "id": "int32"})
        .groupby('postid')
        .agg({'score': 'mean', 'id': 'count'})
        .reset_index()
        .rename(columns={'score': 'avg_comment_score', 'id': 'comment_count'})
    )

    if include_pooled_comments:
        pooled_text = comments.groupby('postid')['text'].apply(lambda x: ' '.join(x)).reset_index()
        pooled_text = pooled_text.rename(columns={'text': 'comments_text'})
        comments_agg = comments_agg.merge(pooled_text, on='postid', how='left')

    if add_tags:
        # 5. One-hot encode Tags
        tags_onehot = tags['tagname'].str.get_dummies()
        tags_onehot['postid'] = tags['id']

    # Generate negative examples
    # Get all unique post IDs
    post_ids = posts['id'].unique()

    # Create a set of existing links for quick lookup
    existing_links = set(zip(post_links['postid'], post_links['relatedpostid']))

    # Generate random pairs of post IDs
    num_negative_samples = len(post_links) * n_negative_samples_per_link  # You can adjust this number
    negative_samples = set()

    while len(negative_samples) < num_negative_samples:
        post1, post2 = np.random.choice(post_ids, 2, replace=False)
        if (post1, post2) not in existing_links and (post2, post1) not in existing_links:
            negative_samples.add((post1, post2))

    # Convert negative samples to DataFrame
    negative_df = pd.DataFrame(list(negative_samples), columns=['postid', 'relatedpostid'])
    negative_df['linktypeid'] = 0  # Label negative examples as 0

    # Label positive examples as 1
    positive_df = post_links.copy()
    positive_df['linktypeid'] = 1

    # Combine positive and negative examples
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Shuffle the dataset
    combined_df = shuffle(combined_df).reset_index(drop=True)

    # Merge features with combined_df
    combined_df = combined_df.merge(posts_features, left_on='postid', right_on='postid', how='left')
    combined_df = combined_df.merge(posts_features, left_on='relatedpostid', right_on='postid', suffixes=('', '_related'), how='left')
    combined_df = combined_df.merge(votes_agg, on='postid', how='left')
    combined_df = combined_df.merge(votes_agg, left_on='relatedpostid', right_on='postid', how='left', suffixes=('', '_related'))
    combined_df = combined_df.merge(comments_agg, on='postid', how='left')
    combined_df = combined_df.merge(comments_agg, left_on='relatedpostid', right_on='postid', how='left', suffixes=('', '_related'))
    if add_tags:
        combined_df = combined_df.merge(tags_onehot, on='postid', how='left')

    combined_df = combined_df[combined_df["body"].isnull() == False]
    combined_df = combined_df[combined_df["body_related"].isnull() == False]

    if add_bertscore:
        combined_df["bertscore_title_f1"] = calculate_bertscore(
            combined_df,
            prediction_column="title",
            reference_column="title_related"
        )
        combined_df["bertscore_body_f1"] = calculate_bertscore(
            combined_df,
            prediction_column="body",
            reference_column="body_related"
        )
        combined_df["bertscore_title_recall"] = calculate_bertscore(
            combined_df,
            prediction_column="body_related",
            reference_column="title",
            metric="recall"
        )
        combined_df["bertscore_title_related_recall"] = calculate_bertscore(
            combined_df,
            prediction_column="body",
            reference_column="title_related",
            metric="recall"
        )

    # Prepare final dataset
    X = combined_df
    y = combined_df['linktypeid']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_data_faiss(data: dict):
    posts = data["Posts"][data["Posts"]["posttypeid"] == "1"]
    posts["body"] = posts["body"].map(clean_html)

    model = SentenceTransformer("BAAI/bge-m3")

    body_embeddings = model.encode(posts["body"].tolist())
    title_embeddings = model.encode(posts["title"].tolist())
    combined_embeddings = np.hstack((body_embeddings, title_embeddings))

    return combined_embeddings

def preprocess_data_faiss_test(data: dict):
    post_links = data["PostLinks"]
    posts = data["Posts"][data["Posts"]["posttypeid"] == "1"]
    posts["id"] = posts["id"].astype("int32")
    posts = posts.set_index("id")
    posts['body'] = posts['body'].map(clean_html)
    posts = posts[["body", "title"]]

    # existing_links = set(zip(post_links['postid'], post_links['relatedpostid']))
    # num_negative_samples = len(post_links)
    # negative_samples = set()
    #
    # while len(negative_samples) < num_negative_samples:
    #     post1, post2 = np.random.choice(posts.index, 2, replace=False)
    #     if (post1, post2) not in existing_links and (post2, post1) not in existing_links:
    #         negative_samples.add((post1, post2))
    #
    # negative_df = pd.DataFrame(list(negative_samples), columns=['postid', 'relatedpostid'])
    # negative_df['linktypeid'] = 0

    positive_df = post_links.copy()
    positive_df['linktypeid'] = 1

    # combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    combined_df = positive_df

    combined_df = shuffle(combined_df).reset_index(drop=True)
    combined_df.drop(columns=["id", "creationdate"], inplace=True)

    combined_df["postid"] = posts.index.get_indexer(combined_df["postid"].astype("int32"))
    combined_df["relatedpostid"] = posts.index.get_indexer(combined_df["relatedpostid"].astype("int32"))
    combined_df = combined_df[(combined_df["postid"] != -1) & (combined_df["relatedpostid"] != -1)]

    return posts, combined_df
