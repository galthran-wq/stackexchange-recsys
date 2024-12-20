# Recommender Systems project
### Authors: Raman Kravets, Leu Marozau

The project's code is available on [GitHub](https://github.com/galthran-wq/stackexchange-recsys).

### Problem Statement

Our goal was to implement some recommender system techniques in order to recommend similar/duplicate questions based on
Stack Exchange public data dump. We also needed to find the way to evaluate the quality of the resulting recommendations.

### Dataset

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

### CatBoost

We have experimented with the CatBoost library to train a model for recommending similar or duplicate questions. Below are the details of the training process, results, and evaluation metrics.

#### Training Process

- **Data Preparation**: We used a combination of features extracted from the posts, including text embeddings for titles and bodies, as well as aggregated features from votes and comments. We used all the positive pairs provided in the dataset and generated randomly sampled negative examples to create a balanced training set.
- **Model Configuration**: The CatBoost model was configured with 1000 iterations, a learning rate of 0.1, and a tree depth of 6. Text features included the body and title of the posts.
- **Training and Evaluation**: The model was trained to predict whether a given pair of posts is relevant. The dataset was split into 80% training and 20% testing data. The training process included monitoring the model's performance on the test set to prevent overfitting.

#### Results

- **Accuracy**: The model achieved an accuracy of 91.74% on the test set.
- **Feature Importance**: The most important features included the body and title embeddings, as well as vote counts.

#### Retrieval Examples

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

#### Evaluation Metrics

The model's performance was evaluated using Recall and NDCG (Normalized Discounted Cumulative Gain) at various cut-off points. Below is a summary of the results:

| Metric       | @5   | @10  | @30  | @50  | @100 |
|--------------|------|------|------|------|------|
| Recall       | 0.25 | 0.37 | 0.56 | 0.64 | 0.76 |
| NDCG         | 0.16 | 0.20 | 0.25 | 0.26 | 0.28 |

These metrics indicate the model's ability to retrieve relevant documents effectively, with higher values representing better performance.

### Faiss

...

### Conclusion

...

