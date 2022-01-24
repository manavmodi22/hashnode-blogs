## Case Study: Data Preprocessing

In the final blog of this series, we will walk through the entire preprocessing workflow on the dataset related to UFO sightings. Each row in this dataset contains information like the location, the type of the sighting, the number of seconds and minutes the sighting lasted, a description of the sighting, and the date of the sighting was recorded.

The actual implementation of this is present in the python notebook [here](https://github.com/manavmodi22/Preprocessing-for-Machine-Learning-in-Python/blob/main/data_preprocessing_casestudy_exercise.ipynb). So it is highly recommended to keep it open while reading through the blog.

Steps:
1. Checking column types
2. Dropping missing data
3. Extracting numbers from strings
4. Identifying features for standardization
5. Encoding categorical variables
6. Features from dates
7. Text Vectorization
8. Selecting the ideal dataset
9. Modeling the dataset.

#### Categorical variables and standardization

Let's tackle the categorical variables and standardization in the UFO dataset.

There are a number of categorical variables in the UFO dataset, including the location data and the type of encounter. These need to be one-hot encoded. 

In addition, we need to standardize the `seconds` column. Check the variance using the `var()` method and log normalize using NumPy's log function.

#### Engineering new features!✨

There are several fields in the UFO dataset that are great candidates for feature engineering. 
+ From the `date` field, we may want to know the month of the sighting. 
+ The number of minutes needs to be extracted from the `length of time` field. 
+ The `description` field contains a text description of the sighting. It would be interesting to vectorize that text and see what we can learn from it.

Some important code to remember for `date` extraction is to use attributes like month and hour to get the pieces of the date you need. 

Regular expressions will help you extract numbers from text, and you can use group to return your results. 

Scikit-learn and tf-idf vectorizer will vectorize text fields.

#### Feature Selection and Modeling

We need to do a little bit of feature selection before we model this data. 

Keep in mind that you want to eliminate redundant features, and there are a couple of candidates for that in this dataset, both in its original form and due to feature engineering. 

We also have a text vector that we can inspect and eliminate words from.

#### Final Thoughts🎓

Remember that `preprocessing` and `modeling` are often iterative practices, and it might take a few tries to find the ideal feature configuration that improves your model's performance. It also helps to be extremely knowledgeable about the dataset that you're working with, as well as having a good understanding of the model you're trying to build.

Check out the exercises linked to this [here](https://github.com/manavmodi22/Preprocessing-for-Machine-Learning-in-Python/blob/main/data_preprocessing_casestudy_exercise.ipynb)
  
Interested in Machine Learning content? Follow me on [Twitter](https://twitter.com/manavmtwt) and [HashNode](https://hashnode.com/@manavmodi0004).







