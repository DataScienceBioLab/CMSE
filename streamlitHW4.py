import streamlit as st
import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot a heatmap of correlations
def plot_correlation_heatmap():
    st.write("Correlation Heatmap:")
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

# Function to plot a heatmap of covariance
def plot_covariance_heatmap():
    st.write("Covariance Heatmap:")
    cov_matrix = df.cov()
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

# Function to plot Alkphos vs MCV by number of drinks
def plot_alkphos_vs_mcv():
    st.write("Alkphos vs MCV by Number of Drinks:")
    sns.scatterplot(x="Alkphos", y="MCV", hue="Drinks", data=df)
    plt.xlabel("Alkphos")
    plt.ylabel("MCV")
    st.pyplot()

# Function to plot SGPT vs SGOT
def plot_sgpt_vs_sgot():
    st.write("SGPT vs SGOT:")
    sns.scatterplot(x="SGPT", y="SGOT", data=df)
    plt.xlabel("SGPT")
    plt.ylabel("SGOT")
    st.pyplot()
    
    # Function to plot a normalized heatmap of covariance
def plot_normalized_covariance_heatmap():
    st.write("Normalized Covariance Heatmap:")
    # Calculate the covariance matrix
    cov_matrix = df.cov()
    
    # Normalize the covariance values
    min_val = cov_matrix.values.min()
    max_val = cov_matrix.values.max()
    normalized_cov_matrix = (cov_matrix - min_val) / (max_val - min_val)
    
    sns.heatmap(normalized_cov_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()


# Function to create a scatterplot with chosen axes, color by number of drinks, and custom symbols for selector
def plot_scatterplot():
    st.write("Scatterplot by Number of Drinks:")
    x_axis = st.selectbox("Select X-axis Feature", df.columns)
    y_axis = st.selectbox("Select Y-axis Feature", df.columns)
    
    # Color points by the number of drinks and symbol by selector
    hue = "Drinks"
    style = "Selector"
    
    # Map selector values to "Diseased" and "Healthy"
    df['Selector'] = df['Selector'].map({1: 'Diseased', 2: 'Healthy'})
    
    # Define custom markers for "Diseased" and "Healthy"
    markers = {"Diseased": "d", "Healthy": "o"}
    
    sns.scatterplot(x=x_axis, y=y_axis, hue=hue, style=style, data=df, palette="viridis", markers=markers)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot()

col1, col2, col3 = st.columns([1, 3, 1])

# Open the zip file
with zipfile.ZipFile('liver+disorders.zip', 'r') as zip_file:
    # Choose a specific file from the archive
    target_file = 'bupa.data'
    
    # Read the content of the chosen file using Pandas and provide corrected column names
    column_names = ["MCV", "Alkphos", "SGPT", "SGOT", "Gammagt", "Drinks", "Selector"]
    df = pd.read_csv(zip_file.open(target_file), header=None, names=column_names, sep=",")

# Display some information in col1 and col2
col1.markdown("This is col1")
col1.markdown("Here's some info in col1")

# Dropdown to select the plot type
selected_plot = col2.selectbox("Select a Plot", ["Correlation Heatmap", "Covariance Heatmap", "Alkphos vs MCV by Number of Drinks", "SGPT vs SGOT", "Normalized Covariance Heatmap"])

# Display the selected plot
if selected_plot == "Correlation Heatmap":
    plot_correlation_heatmap()
elif selected_plot == "Covariance Heatmap":
    plot_covariance_heatmap()
elif selected_plot == "Alkphos vs MCV by Number of Drinks":
    plot_alkphos_vs_mcv()
elif selected_plot == "SGPT vs SGOT":
    plot_sgpt_vs_sgot()
elif selected_plot == "Normalized Covariance Heatmap" :
    plot_normalized_covariance_heatmap()


plot_scatterplot()

# You can continue to add more content to col3 or make other modifications as needed
col3.markdown("This is col3")
col3.markdown("You can add more content here.")