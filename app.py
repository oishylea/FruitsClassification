import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib import patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn import neighbors

# Load the dataset
fruits = pd.read_table('fruit_data_with_colors.txt')

# Streamlit app title
st.title("Fruit Classification 🍓🍉🍒🍑")
st.write("Classifying different types of fruits based on various attributes.")

# Create two columns with different widths
col1, col2 = st.columns([2, 1])  # 2:1 ratio for column widths

# Part 1: Display dataset and its shape
with col1:
    st.subheader("Dataset Preview")
    st.write(fruits.head())
    st.write("Shape of the dataset:", fruits.shape)

# Part 2: Display unique fruit names
with col2:
    st.subheader("Fruit Names")
    st.write(fruits['fruit_name'].unique())



col3, col4 = st.columns(2) 

with col3:

    # Display counts of each fruit
    st.subheader("Fruit Counts")
    fruit_counts = fruits.groupby('fruit_name').size()
    st.bar_chart(fruit_counts)

    # Display the KNN plot
    st.subheader("KNN Decision Boundary")
    st.image('knn.png')

with col4:
    st.subheader("Box Plot")
    st.image('fruits_boxfig.png')

    st.subheader("Scatter-Matrix")
    st.image('scatter.png')