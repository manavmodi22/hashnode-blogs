## What is Data Standardizing?

 #### What is Data Standardization?

It is a preprocessing method used to transform continuous data to make it look normally distributed. 

**Why?**

Scikit-learn models assume normally distributed data. In the case of continuous data, we might risk biasing the models.

Two methods can be used for the standardization process:
1. Log Normalization
2. Feature Scaling

These methods are applied to continuous numerical data.

**When?**

1. Models are present in linear space. (Ex. KNN, KMeans, etc.),data must also be in linear space.
2. Dataset features that have high variance. This could bias a model that assumes it is normally distributed. 
3. Modeling dataset that has features that are continuous and on different scales. 
> For example, a dataset that has height and weight as its features needs to be standardized to make sure they are on the same linear scale.

#### What is Log Normalization?

1. Log transformation is applied
2. Used in datasets where the variance of a particular column is significantly high as compared to other columns
3. Natural log is applied on values

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642770484282/TTQ1ZEYkw.png)

4. It is used to captured relative changes, and magnitude of change, and keeps everything in the positive space.

Let's see the implementation.

``` 
print(df) 
```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642770726107/Y5bhqEDVp.png)

``` 
print(df.var())
```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642770738314/DqEMoYHbC.png)

We will use the log operator from the NumPy library to perform the normalization.

``` 
import numpy as np
df["log_2"] = np.log(df["col2"])
print(df)

```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642770768558/J8SBzLm-V.png)
Let's see values.

```
print(np.var(df[["col1","log_2"]]))

```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642770825749/mNxa5rRZU5.png)


#### What is feature scaling?

This method is useful when 
1. continuous features are present on different scales. 
2. model is in linear scale.

The transformation on the dataset is done such that the resultant mean is 0 and the variance is 1.

Here across the features, you can see how the variation is.

```
print(df)
```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642776382839/PIOIrDxDH.png)
```
print(df.var())
```
![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642776418943/lR2Su08Lj.png)

Using the standardscaler method from sklearn, the process is done.

```
from sklearn.preprocessing  import StandardScaler
scaler = StandardScaler()
df_scaled= pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_scaled)
print(df.var())

```

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642776443908/FBChoCtcn.png)

![image.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1642776460766/Fm6fv4aib.png)

Check out the exercises linked to this [here](https://github.com/manavmodi22/Preprocessing-for-Machine-Learning-in-Python/blob/main/data_preprocessing_chapter2_exercise.ipynb)
  
Interested in Machine Learning content? Follow me on [Twitter](https://twitter.com/manavmtwt) and [HashNode](https://hashnode.com/@manavmodi0004).




