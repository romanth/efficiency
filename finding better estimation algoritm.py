#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris=load_iris()
X=iris.data
y=iris.target


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
#from sklearn.neighbors import KNeighborsClassifier


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[22]:


# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)


# In[23]:


print(y_train.shape)
print(y_test.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[26]:


y_pred = logreg.predict(X_test)


# In[27]:


from sklearn import metrics


# In[28]:


print(metrics.accuracy_score(y_test, y_pred))


# In[30]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[31]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[33]:


# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[34]:


import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[ ]:




