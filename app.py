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

    # Prepare the data
    X = fruits[['mass', 'width', 'height', 'color_score']]
    y = fruits['fruit_label']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    def plot_fruit_knn(X, y, n_neighbors, weights):
        X_mat = X[['height', 'width']].to_numpy()  
        y_mat = y.to_numpy()

        # Create color maps
        cmap_light = ListedColormap(['#E41100', '#FF8A89', '#FF7F1A', '#FFE785'])
        cmap_bold = ListedColormap(['#E41100', '#FF8A89', '#FF7F1A', '#FFE785'])

        # Initialize and train the KNN classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_mat, y_mat)

        # Plot decision boundary
        mesh_step_size = .01  # Step size in the mesh
        plot_symbol_size = 50

        x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
        y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                            np.arange(y_min, y_max, mesh_step_size))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Create legend
        patch0 = mpatches.Patch(color='#E41100', label='Apple')
        patch1 = mpatches.Patch(color='#FF8A89', label='Banana')
        patch2 = mpatches.Patch(color='#FF7F1A', label='Orange')
        patch3 = mpatches.Patch(color='#FFE785', label='Grape')


        # Show the plot in Streamlit
        st.pyplot(plt)
        plt.close()  # Close the figure to avoid display issues

    # Display the KNN plot
    st.subheader("KNN Decision Boundary")
    plot_fruit_knn(X_train, y_train, 5, 'uniform')

with col4:
    # Optional: Create a scatter plot
    st.subheader("Scatter Plot of Fruit Dimensions")
    fig, ax = plt.subplots()
    ax.scatter(fruits['width'], fruits['height'], c=fruits['color_score'], cmap='viridis')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Fruit Dimensions')
    st.pyplot(fig)

    # Create box plots
    st.subheader("Box Plot For Each Input Variable")

    # Drop 'fruit_label' and create box plots for numeric columns
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2, 2), 
                                            sharex=False, sharey=False, ax=axs.flatten())

    # Set the title for the plot
    plt.suptitle('Box Plot For Each Input Variable')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    # Save the figure to a file and display it
    plt.savefig('fruits_boxfig.png')
    st.pyplot(fig) 