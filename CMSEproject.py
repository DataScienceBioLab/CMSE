#streamlit project
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, pearsonr


# Fetch and preprocess the data
@st.cache  # 👈 This function will be cached
def load_data():
    # Concatenate features and targets into a single DataFrame
    df = pd.read_csv('heart_disease_uci.csv', nrows=303)
    # Your preprocessing logic here
    # after renaming the columns, I will be dealing with the Nans
    # I am dropping the num_fluoro column, as it is reduntant to art_block
    new_column_names = [
    'age', 'is_male', 'chest_pain', 'rest_bp', 'chol',
    'high_sugar', 'rest_ecg', 'max_hr', 'exercise_angina',
    'st_depression', 'st_slope', 'num_fluoro',
    'thalass_type', 'art_blocks'
    ]
    df.columns = new_column_names
    
    df = df.drop('num_fluoro', axis = 1) # this is an effective duplicate of blockages
    df.at[87, 'thalass_type'] = 3.0 # patient 87 has no art blocks, went with mode for thalass when block == 0
    df.at[266, 'thalass_type'] = 7.0 # for this patient I compared other patietnts with blockage and exercise angiana and grabbed the mode


    return df

# Load your data
data = load_data()

# Title of the app
st.title("Heart Disease Data Explorer")

# A brief description
st.write("This app enables basic exploratory analysis of the heart disease dataset.")

# Showing the dataset
st.subheader("Dataset")
st.write(data.head(303))

# Creating a sidebar for user inputs:
st.sidebar.header('User Input Parameters')

# Create a slider in the sidebar:
feature = st.sidebar.selectbox(
    'Select a feature to visualize',
    ('age', 'is_male', 'chest_pain', 'rest_bp', 'chol')
)

# Visualizations based on user input:
st.subheader(f'Distribution of {feature}')
fig, ax = plt.subplots(figsize=(10, 3))
sns.histplot(data[feature], kde=False, bins=30)
st.pyplot(fig)  # Plot will be shown in Streamlit app

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)


# Assuming df is your DataFrame with all the variables
df = load_data()
# Your categorizations here
numeric_vars = ['age', 'rest_bp', 'chol', 'max_hr', 'st_depression']
binary_vars = ['is_male', 'high_sugar', 'exercise_angina']
multi_cat_vars = ['chest_pain', 'rest_ecg', 'st_slope', 'thalass_type', 'art_blocks']
df[numeric_vars] = df[numeric_vars].apply(lambda x: (x - x.mean()) / x.std())


# Placeholder for the correlation matrix and type matrix
corr_matrix = pd.DataFrame(np.zeros((len(data.columns), len(data.columns))), columns=data.columns, index=data.columns)
corr_type_matrix = corr_matrix.copy().astype(str)  

for var1 in data.columns:
    for var2 in data.columns:
        # Binary variables
        if var1 in binary_vars and var2 in binary_vars:
            corr, _ = spearmanr(data[var1], data[var2])
            corr_type = "S"  # Spearman
        # Continuous variables
        elif var1 in numeric_vars and var2 in numeric_vars:
            corr, _ = pearsonr(data[var1], data[var2])
            corr_type = "P"  # Pearson
        # Categorical variables
        elif var1 in multi_cat_vars and var2 in multi_cat_vars:
            chi2, _, _, _ = chi2_contingency(pd.crosstab(data[var1], data[var2]))
            corr = np.sqrt(chi2 / (chi2 + data.shape[0]))
            corr_type = "C"  # Chi-square
        # Other variable types
        else:
            corr, _ = pearsonr(data[var1], data[var2])
            corr_type = "P"  # Pearson
        
        corr_matrix.loc[var1, var2] = corr
        corr_type_matrix.loc[var1, var2] = corr_type

# Streamlit Display
st.title('Heart Disease Exploratory Analysis')
st.write('Correlation heatmaps representing relationships and correlation types between variables.')

# Plotting the Correlation Values
fig, ax = plt.subplots(figsize=(12, 10))  # Adjusted size for readability
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Coefficients")
st.pyplot(fig)

# Plotting the Correlation Types
fig, ax = plt.subplots(figsize=(12, 4))  # Adjusted size for readability
sns.heatmap(pd.DataFrame(np.zeros(corr_type_matrix.shape), columns=data.columns, index=data.columns), annot=corr_type_matrix, fmt="", cmap="Blues", cbar=False, ax=ax)
plt.title("Correlation Types")
st.pyplot(fig)

# Adding a legend
st.write('Correlation Types: S = Spearman, P = Pearson, C = Chi-square')

# Streamlit app
st.title('Correlation Heatmap of Variables')
st.write('This is a sample heatmap of correlations between variables of different types.')

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
st.pyplot(plt)


# Sidebar widgets for scatter plot variables
st.sidebar.header('Scatter Plot Parameters')
scatter_x = st.sidebar.selectbox('Select variable for x', df.columns, index=0)
scatter_y = st.sidebar.selectbox('Select variable for y', df.columns, index=1)

# Scatter plot
st.subheader(f"Scatter Plot of {scatter_x} vs {scatter_y}")
plt.figure(figsize=(10, 6))

# Using seaborn to create the scatter plot
sns.scatterplot(x=scatter_x, y=scatter_y, hue="art_blocks", style="is_male", data=data, palette="viridis", markers=["o", "s"])
plt.title(f"{scatter_x} vs {scatter_y} by Art Blocks and Gender")
st.pyplot(plt)

# Explanation of plot
st.write(f"This scatter plot shows {scatter_x} on the x-axis and {scatter_y} on the y-axis. Points are colored by 'art_blocks' and shaped by 'is_male' gender status (square for male, circle for female).")


# Sidebar widgets for violin plot variable
st.sidebar.header('Violin Plot Parameters')
violin_var = st.sidebar.selectbox('Select variable for Violin Plot', df.columns, index=0)


# Subheader
st.subheader("Distributions of Variables")

# Violin Plots for all numeric variables
st.markdown("### Violin Plots for Numeric Variables")
fig, axs = plt.subplots(nrows=1, ncols=len(numeric_vars), figsize=(15, 4), sharey=False)
for i, var in enumerate(numeric_vars):
    sns.violinplot(x=df[var], ax=axs[i])
    axs[i].set_title(f"{var}")
plt.tight_layout()
st.pyplot(fig)
st.write("Violin plots represent the distribution of numeric variables. The width of the plot at any given y-value indicates the density of values at that level.")

# Box Plots for all categorical variables
all_categorical_vars = binary_vars + multi_cat_vars  # combining both categorical var lists
st.markdown("### Boxplots for Categorical Variables")
fig, axs = plt.subplots(nrows=1, ncols=len(all_categorical_vars), figsize=(15, 4), sharey=False)
for i, var in enumerate(all_categorical_vars):
    sns.boxplot(x=df[var], ax=axs[i])
    axs[i].set_title(f"{var}")
plt.tight_layout()
st.pyplot(fig)
st.write("Boxplots provide a summary of the categorical variables. They visually display the minimum, first quartile, median, third quartile, and maximum of the datasets.")
