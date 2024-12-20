# Recommender Systems project
### Authors: Raman Kravets, Leu Marozau

The project's code is available on [GitHub](https://github.com/galthran-wq/stackexchange-recsys).

## Problem Statement

Our goal was to implement some recommender system techniques in order to recommend similar/duplicate questions based on
Stack Exchange public data dump. We also needed to find the way to evaluate the quality of the resulting recommendations.

## Dataset

Stack Exchange publishes its data dumps every three months. The file dump contains info about all Stack Exchange sites
(including famous Stack Overflow). Database scheme documentation is available [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede).

The archive with the most recent data dump has file size of 90 GB (with Stack Overflow alone – 63.3 GB).
We decided that we would not be able to process such a large dataset on our home computers.
First of all, we searched for older data dumps of Stack Overflow, but they were still very big.
That is why we decided to use data from [another Stack Exchange website: focused on questions about cooking](https://cooking.stackexchange.com).
We used data from [March 2015 dump](https://archive.org/details/stackexchange_20150313).
The size of the archive is 29.5M, the size of the unpacked data is 153 MB.
The database schema here is the same as in Stack Overflow.

This dataset contains 11812 questions with different info such as title, question text, tags, view count, rating, etc.

The dataset also has 4272 post links that match pairs of related questions. Some of them match deleted questions,
so we deleted them from the dataset. The resulting amount of post links is 3717.

## CatBoost

We have experimented with the CatBoost library to train a model for recommending similar or duplicate questions. Below are the details of the training process, results, and evaluation metrics.

### Training Process

- **Data Preparation**: We used a combination of features extracted from the posts, including text embeddings for titles and bodies, as well as aggregated features from votes and comments. We used all the positive pairs provided in the dataset and generated randomly sampled negative examples to create a balanced training set.
- **BERTScore**: We experimented with BERTScore metric as one of the features. This metrics computates similarity between two texts based on token embeddings. We experimented with different setups (f1-score/recall, title-title/body-body/title-body). It showed really good predictive power, but had a massive flaw: high computational cost. That was the reason why we had to get rid of it.
- **Model Configuration**: The CatBoost model was configured with 1000 iterations, a learning rate of 0.1, and a tree depth of 6. Text features included the body and title of the posts.
- **Training and Evaluation**: The model was trained to predict whether a given pair of posts is relevant. The dataset was split into 80% training and 20% testing data. The training process included monitoring the model's performance on the test set to prevent overfitting.

### Results

- **Accuracy**: The model achieved an accuracy of 91.74% on the test set.
- **Feature Importance**: The most important features included the body and title embeddings, as well as vote counts.

Classification report:

```
              precision    recall  f1-score   support

    Not Related       0.92      0.92      0.92      1858
    Related           0.92      0.92      0.92      1858

    accuracy                           0.92      3716
    macro avg       0.92      0.92      0.92      3716
    weighted avg    0.92      0.92      0.92      3716
```

### Retrieval Examples

The model was used to retrieve the top 5 closest documents for a given test post. Here are some examples (only titles are shown):

- **Test Post ID: 10042**
  - **Title**: How to make softer biscuits?
  - **Top 5 Closest Documents**:
    1. Translating cooking terms between US / UK / AU / CA / NZ
    2. Difference between Maida and All purpose flour
    3. Butter substitute for 1 cup of butter for baking
    4. What is the purpose of sifting dry ingredients?
    5. Are there any general principles of ingredient substitutions?

- **Test Post ID: 10070**
  - **Title**: 1/4 cup of shredded basil OR 1/4 cup of basil that is then shredded?
  - **Top 5 Closest Documents**:
    1. What is a good use for lots of fresh cilantro?
    2. Translating cooking terms between US / UK / AU / CA / NZ
    3. Is it worth tearing lettuce for salad?
    4. When, if ever, are dried herbs preferable to fresh herbs?
    5. How can you reduce the heat of a chili pepper?

- **Test Post ID: 10280**
  - **Title**: What can I do with frozen eggs?
  - **Top 5 Closest Documents**:
    1. Are refrigerated hard boiled eggs really unsafe after a week?
    2. Should I refrigerate eggs?
    3. How long can I keep eggs in the refrigerator?
    4. Is it safe to eat raw eggs?
    5. How long can eggs be unrefrigerated before becoming unsafe to eat?

- **Test Post ID: 10315**
  - **Title**: How can I ensure food safety if my cooking utensils have touched raw meat?
  - **Top 5 Closest Documents**:
    1. Are there books describing the general principles of cooking?
    2. Translating cooking terms between US / UK / AU / CA / NZ
    3. Why does my food turn out poorly using an All-Clad Stainless-Steel Fry Pan?
    4. How can brown stains be removed from pots and pans?
    5. Difference in technique for cooking with non-stick and standard pans?

### Evaluation Metrics

The model's performance was evaluated using Recall and NDCG (Normalized Discounted Cumulative Gain) at various cut-off points. Below is a summary of the results:

| Metric       | @5   | @10  | @30  | @50  | @100 |
|--------------|------|------|------|------|------|
| Recall       | 0.25 | 0.37 | 0.56 | 0.64 | 0.76 |
| NDCG         | 0.16 | 0.20 | 0.25 | 0.26 | 0.28 |

These metrics indicate the model's ability to retrieve relevant documents effectively, with higher values representing better performance.

## Faiss

In addition to CatBoost, we utilized Faiss library (owned by Meta, a company recognized as extremist in Russia).
It is used for efficient similarity search and clustering of dense vectors.

### Methodology

- **Data Preparation**: We used sentence embeddings for both the body and title of the posts.
These embeddings were generated using [BGE-M3 text embedding model](https://huggingface.co/BAAI/bge-m3) and then combined to form a single vector for each post.
- **Dimensionality Reduction**: To improve the efficiency of the search, we reduced the dimensionality of the embeddings from 2048 to 128 using Principal Component Analysis (PCA).
- **Indexing**: We experimented with two types of indices for similarity search:
  - **IndexFlatL2**: A simple index that performs a brute-force search over the dataset.
  - **IndexLSH**: A locality-sensitive hashing index that provides faster search times at the cost of some accuracy.
- **Search and Retrieval**: For each test post, we retrieved the top k most similar posts based on the L2 distance between their embeddings.

### Results

- **Recall@k**: The recall metric was used to evaluate the performance of the Faiss indices.
We used "post links" table from the dataset for this evaluation.
Below are the recall scores for different values of k:

| Index Type           | @5   | @10  | @15  | @20  |
|----------------------|------|------|------|------|
| IndexFlatL2          | 0.86 | 0.94 | 0.96 | 0.97 |
| IndexFlatL2 with PCA | 0.88 | 0.94 | 0.97 | 0.98 |
| IndexLSH with PCA    | 0.85 | 0.94 | 0.97 | 0.99 |

These results demonstrate high effectiveness of this method in retrieving relevant posts.

Applying PCA allowed us to heavily reduce embedding dimensions but didn't change the result
(the results actually became a little bit better, but the difference is not that substantial to make conclusions about the effectiveness of PCA in this case).

While we specifically searched for the dataset that is quite small and computational speed won't be an issue,
we still decided to experiment with IndexLSH. Theoretically, it should make the search faster at the cost of accuracy.
We experimented with a few n_bits values and it seems that this index produces the results similar to IndexFlatL2 with n_bits = 32+.

The outputs differ a bit when we use different indexes (see the examples in the next part).

### Retrieval Examples

The Faiss method was used to retrieve the top 5 closest documents for a given test post. Here are some examples (only titles are shown):

**Original Post Title**: cooking oven temperature

- **IndexFlatL2**:
  1. How hot is your oven? *(distance = 0.347)*
  2. Right baking temperature *(distance = 0.377)*
  3. The Warm Oven Temp. for Cakes *(distance = 0.380)*
  4. Electric oven Temperature *(distance = 0.407)*
  5. Lowering oven temps *(distance = 0.436)*

- **IndexFlatL2 (with PCA)**:
  1. How hot is your oven? *(distance = 0.352)*
  2. Right baking temperature *(distance = 0.375)*
  3. The Warm Oven Temp. for Cakes *(distance = 0.395)*
  4. Electric oven Temperature *(distance = 0.415)*
  5. Lowering oven temps *(distance = 0.436)*

- **IndexLSH (with PCA, n_bits=64)**:
  1. Right baking temperature *(distance = 13)*
  2. Thermometers for high temperature ovens *(distance = 14)*
  3. What temperature to cook a pork tenderloin at? *(distance = 14)*
  4. What features do I want in a Toaster Oven? *(distance = 14)*
  5. Kitchen essentials for a poor college student who wants to cook like an iron chef? *(distance = 15)*

We can see that IndexLSH recommendations are related, but not as similar as IndexFlatL2 recommendations.


## Conclusion

In this project we applied two methods for building a recommender system for Stack Exchange data dump and evaluated
different metrics to measure the effectiveness of these models.

BERTScore metric showed it's high effectiveness in comparing texts (finding similarities between questions and question titles),
but didn't really suite this particular task because of it's computational cost.

While we worked with a small subset of the dump (data only from cooking.stackexchange.com), the same methodology
can be applied to other Stack Exchange websites (such as Stack Overflow). The data structure is the same across these sites,
but applying the same methods to more popular sites with bigger database may require much higher computational costs.

