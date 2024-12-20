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

