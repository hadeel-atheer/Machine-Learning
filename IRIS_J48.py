
# coding: utf-8

# In[1]:

#import all the necessary libraries and functions
from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[2]:

#read the excel file
df = pd.read_excel('dataset.xlsx')
print (df.columns)


# In[3]:

#checking the dataset values by using specific functions
print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")


# In[4]:

#print the types of class under Iris
print("* iris types:", df["Class"].unique(), sep="\n")


# In[5]:

#PRE-PROCESSING

#changing the names or string values to integers as sklearn works on numeric data and storing in a new column
#this function returns the modified dataframe

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


# In[6]:

# checking the respective number assigned to each class

df2, targets = encode_target(df, "Class")
print("* df2.head()", df2[["Target", "Class"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df2[["Target", "Class"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")

#as can be seen Iris-setosa = 0, iris-virgica = 2


# In[7]:

#creating a list having the names of 4 attributes
features = list(df2.columns[:4])
print("* features:", features, sep="\n")


# In[8]:

#assigning y and x values for a fit. 
#min_samples_split = 20 It means that at least 20 samples will be used to create a split in a tree
#random_state= 99, to generate seed so that results are reproducible
y = df2["Target"]
X = df2[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)


# In[9]:

#VISUALIZATION of the tree

#A function is created that calls 'dot' function to create a png image.
#Graphviz needs to be installed separately also for system
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


# In[10]:

#to visualize it
#the tree will be automatically saved in the working directory when this file is executed
visualize_tree(dt, features)


# In[ ]:



