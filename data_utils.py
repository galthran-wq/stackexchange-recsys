import os


import pandas as pd
import py7zr
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from evaluate import load
from tqdm.notebook import tqdm


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


def preprocess_data(
    data: dict,
    n_negative_samples_per_link: int = 1,
    add_tags: bool = False,
    add_post_text: bool = False,
    include_pooled_comments: bool = False,
    add_bertscore: bool = False
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
        posts_features.append('body')
    posts_features = posts[posts_features]
    if add_post_text:
        posts_features['body'] = posts_features['body'].map(clean_html)
    posts_features = posts_features.rename(columns={'id': 'postid'})

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

    combined_df.drop(columns=["postid", "relatedpostid", "id", "creationdate", "postid_related"], inplace=True)
    combined_df = combined_df[combined_df["body"].isnull() == False]
    combined_df = combined_df[combined_df["body_related"].isnull() == False]

    if add_bertscore:
        bertscore = load("bertscore")

        batch_size = 50
        num_batches = len(combined_df) // batch_size + (len(combined_df) % batch_size > 0)

        bertscore_f1 = []

        for i in tqdm(range(num_batches), desc="Calculating BERTScore"):
            start = i * batch_size
            end = start + batch_size
            batch_predictions = combined_df["body"].iloc[start:end].tolist()
            batch_references = combined_df["body_related"].iloc[start:end].tolist()
            try:
                results = bertscore.compute(
                    predictions=batch_predictions,
                    references=batch_references,
                    lang="en",
                    # model_type = "distilbert-base-uncased",
                    device="cuda"
                )
            except Exception as e:
                print(batch_predictions)
                print("-------------------")
                print(batch_references)
                raise e
            bertscore_f1.extend(results["f1"])

        combined_df["bertscore_f1"] = bertscore_f1


    # Prepare final dataset
    X = combined_df
    y = combined_df['linktypeid']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
