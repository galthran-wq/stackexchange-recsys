import os

import pandas as pd
import py7zr
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def extract_data(data_dir: str):
    with py7zr.SevenZipFile(os.path.join(data_dir, "cooking.stackexchange.com.7z"), mode='r') as archive:
        archive.extractall(path=data_dir)


def load_data(data_dir: str):
    data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".xml"):
            with open(os.path.join(data_dir, filename), "r") as file:
                soup = BeautifulSoup(file, "lxml")
                attributes = [row.attrs for row in soup.find_all("row")]
                data[filename[:-4]] = pd.DataFrame(attributes)
    return data



def preprocess_data(data: dict):
    post_links = data["PostLinks"]
    posts = data["Posts"]
    users = data["Users"]
    votes = data["Votes"]
    tags = data["Tags"]
    comments = data["Comments"]

    # Split data into train and test before preprocessing
    train_post_links, test_post_links = train_test_split(post_links, test_size=0.2, random_state=42)

    # Feature Engineering
    # 1. Extract features from Posts
    posts_features = posts[['id', 'score', 'viewcount', 'answercount', 'commentcount']]
    posts_features = posts_features.rename(columns={'id': 'postid'})

    # 2. Extract features from Users
    users_features = users[['id', 'reputation', 'views', 'upvotes', 'downvotes']]
    users_features = users_features.rename(columns={'id': 'userid'})

    # 3. Aggregate Votes
    votes_agg = votes.groupby('postid').agg({'votetypeid': 'count'}).reset_index()
    votes_agg = votes_agg.rename(columns={'votetypeid': 'vote_count'})

    # 4. Aggregate Comments
    comments_agg = comments.groupby('postid').agg({'score': 'mean', 'id': 'count'}).reset_index()
    comments_agg = comments_agg.rename(columns={'score': 'avg_comment_score', 'id': 'comment_count'})

    # 5. One-hot encode Tags
    tags_onehot = tags['tagname'].str.get_dummies()
    tags_onehot['postid'] = tags['id']

    # Merge features with PostLinks for train data
    train_post_links = train_post_links.merge(posts_features, left_on='postid', right_on='postid', how='left')
    train_post_links = train_post_links.merge(posts_features, left_on='relatedpostid', right_on='postid', suffixes=('', '_related'), how='left')
    train_post_links = train_post_links.merge(votes_agg, on='postid', how='left')
    train_post_links = train_post_links.merge(comments_agg, on='postid', how='left')
    train_post_links = train_post_links.merge(tags_onehot, on='postid', how='left')

    X_train = train_post_links
    y_train = train_post_links['linktypeid']

    # Preprocess test data
    test_post_links = test_post_links.merge(posts_features, left_on='postid', right_on='postid', how='left')
    test_post_links = test_post_links.merge(posts_features, left_on='relatedpostid', right_on='postid', suffixes=('', '_related'), how='left')
    test_post_links = test_post_links.merge(votes_agg, on='postid', how='left')
    test_post_links = test_post_links.merge(comments_agg, on='postid', how='left')
    test_post_links = test_post_links.merge(tags_onehot, on='postid', how='left')

    X_test = test_post_links
    y_test = test_post_links['linktypeid']

    return X_train, X_test, y_train, y_test
