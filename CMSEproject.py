#streamlit project
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from scipy.stats import zscore

# Fetch and preprocess the data
@st.cache  # 👈 This function will be cached
def load_data():
        """
    Load and preprocess the heart disease dataset.
    
    - Fetch the dataset from the UCI ML Repository
    - Rename columns for clarity
    - Handle missing data
    - Drop redundant columns
    
    Returns: 
    df: DataFrame - Preprocessed data
    """
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

def calculate_correlations(var1, var2, data, binary_vars, numeric_vars, multi_cat_vars):
    """
    Calculate the statistical correlation between two variables using different methods
    based on their type (binary, numeric, or multi-category).
    
    Parameters:
    - var1, var2: Variables for correlation calculation
    - data: DataFrame - The dataset
    - binary_vars, numeric_vars, multi_cat_vars: Lists - Variables categorized by type
    
    Returns:
    - corr: float - Calculated correlation coefficient
    - corr_type: str - Type of correlation coefficient ("S": Spearman, "P": Pearson, "C": Chi-square)
    """
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
    return corr, corr_type

# Load the preprocessed data
data = load_data()

# Introduce the dataset to the Streamlit user
st.header("Data Overview")
st.write("""
This dataset provides a wide variety of medical attributes, aiming to predict 
the presence of heart disease. For our exploration, we particularly focus on 14 
key attributes, exploring their relationship and impact on the diagnosis of 
heart disease. These attributes range from demographic information, such as 
age and sex, to more specific medical indicators, like cholesterol levels, 
resting blood pressure, and different types of test results.
""")


# Descriptions of selected variables
selected_vars_desc = {
    'age': "Age in years.",
    'is_male': "Sex of the individual (1 = male; 0 = female).",
    'chest_pain': """Type of chest pain experienced:
        - Value 1: Typical angina
        - Value 2: Atypical angina
        - Value 3: Non-anginal pain
        - Value 4: Asymptomatic""",
    'rest_bp': "Resting blood pressure (in mm Hg upon hospital admission).",
    'chol': "Serum cholesterol level (in mg/dl).",
    'high_sugar': "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).",
    'rest_ecg': """Resting electrocardiographic results:
        - Value 0: Normal
        - Value 1: ST-T wave abnormality
        - Value 2: Probable or definite left ventricular hypertrophy""",
    'max_hr': "Maximum heart rate achieved during a stress test.",
    'exercise_angina': "Exercise-induced angina (1 = yes; 0 = no).",
    'st_depression': "ST depression induced by exercise relative to rest.",
    'st_slope': """The slope of the peak exercise ST segment:
        - Value 1: Upsloping
        - Value 2: Flat
        - Value 3: Downsloping""",
    'num_fluoro': "Number of major vessels colored by fluoroscopy (0-3).",
    'thalass_type': """Thalassemia type:
        - Value 3: Normal
        - Value 6: Fixed defect
        - Value 7: Reversible defect""",
    'art_blocks': "Diagnosis of heart disease (>50% diameter narrowing: 1 = yes; 0 = no)."
}

# Displaying Descriptive Information about Variables to the User
st.header("Variable Descriptions")
# Iterating through the predefined variable descriptions and displaying them
for var, desc in selected_vars_desc.items():
    st.subheader(var)  # Variable name as subheader
    st.write(desc)  # Description of the variable

# Categorization of Variables for Analytical Reference
numeric_vars = ['age', 'rest_bp', 'chol', 'max_hr', 'st_depression']  # Continuous variables
binary_vars = ['is_male', 'high_sugar', 'exercise_angina']  # Binary variables
multi_cat_vars = ['chest_pain', 'rest_ecg', 'st_slope', 'thalass_type', 'art_blocks']  # Categorical variables

# Combining Variable Types for General Reference
cat_vars = binary_vars + multi_cat_vars  # All categorical variables
all_vars = numeric_vars + binary_vars + multi_cat_vars  # All variables

# Providing Title and Brief Description for the App
st.title("Heart Disease Data Explorer")
st.write("This app enables basic exploratory analysis of the heart disease dataset.")

# Data Display Section
# Implementing a toggle to allow users to view the raw dataset
if st.checkbox("Show Raw Data"):
    st.subheader("Dataset")
    st.write(data.head())

# Providing Additional Comments and Context about the Dataset
st.markdown("""### Comments on Dataset
This dataset, sourced from the UCI ML Repo, contains 303 observations across 14 features. 
One feature, `num_fluoro`, was removed due to redundancy and substantial missing data. The primary aim is to investigate predictors of arterial blockages, utilizing the remaining features.
""")

# Visualization Section
st.subheader(f'Hiplot for Parallel Coordinates')

# Initialize subplots to display visualizations related to the 'art_blocks' variable
f, (a1, a2) = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))

# Creating Parallel Coordinate Plot with Original Data for 'art_blocks'
# This visualization helps identify patterns and trends related to arterial blockages
pd.plotting.parallel_coordinates(data, "art_blocks", ax=a1)
a1.set_title("Parallel Coordinates for Arterial Blockages Without zscore")

# Normalizing numerical columns for effective visualization and plotting with 'art_blocks'
# This z-score normalization helps understand the relative position of each variable
numeric_cols = data.select_dtypes(include=[np.number]).columns
df_s = data[numeric_vars].apply(zscore)  # Apply z-score normalization
df_s[cat_vars] = data[cat_vars]  # Append categorical variables without normalization
pd.plotting.parallel_coordinates(df_s, "art_blocks", ax=a2)
a2.set_title("Parallel Coordinates for Arterial Blockages With zscore")
# Display the matplotlib figure in the Streamlit app
st.pyplot(f)

# Violin Plots of Numeric Variables
st.subheader('Violin Plots for Numeric Variables')

# Set up the matplotlib figure with 4 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Specify variable groups
# These are the variables for which violin plots will be generated
group1 = ['chol']
group2 = ['age']
group3 = ['st_depression']
group4 = ['rest_bp', 'max_hr']

# Prepare data for the violin plot and plot
# Loop through the groups and corresponding axes to create plots
for group, ax in zip([group1, group2, group3, group4], [ax1, ax2, ax3, ax4]):
    # Melt the data to long format, which is suitable for seaborn
    data_melted = pd.melt(data, id_vars='art_blocks', value_vars=group)
    
    # Creating a violin plot using seaborn. Hue is used to differentiate data based on 'art_blocks'
    sns.violinplot(x='variable', y='value', hue='art_blocks', data=data_melted,
                   split=False, inner="quart", linewidth=1.3, palette="muted", ax=ax)
    
    # Customize the plot appearance
    # Set title, x and y axis labels, and legend details
    ax.set_title('Violin Plot of ' + ', '.join(group) + ' by Arterial Blockages')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Value')
    ax.legend(title='Arterial Blockages', title_fontsize='13', fontsize='11')

# Adjust the layout to avoid overlap
plt.tight_layout()

# Display the plot in the Streamlit app
st.pyplot(fig)

# Creating a sidebar for user inputs:
# Introducing a header for this section in the sidebar
st.sidebar.header('User Input Parameters')

# Create a slider in the sidebar:
# Enabling the user to select a feature to visualize through a dropdown menu
feature = st.sidebar.selectbox(
    'Select a categorical variable to visualize with a histogram',
    all_vars  # List containing all variable names
)

# Visualizations based on user input:
# Displaying a subheader that dynamically updates based on the user-selected feature
st.subheader(f'Distribution of {feature}')

# Optional: User input for bin width in the histogram
# Enabling the user to select the bin width of the histogram via a slider
bin_width = st.sidebar.slider('Select bin width for the histogram', min_value=5, max_value=100, value=20)

# Creating a figure and axis object to plot the histogram using Seaborn
fig, ax = plt.subplots(figsize=(6, 4))

# Plotting a histogram of the selected feature with the specified bin width
sns.histplot(data[feature], bins=bin_width, kde=False)  # kde=False means no density curve will be displayed
plt.xlabel(feature)  # Labeling the x-axis with the selected feature name
plt.ylabel('Count')  # Labeling the y-axis as 'Count'
plt.title(f'Distribution of {feature} in the dataset')  # Adding a title to the plot

# Displaying the plot in the Streamlit app
st.pyplot(fig)

# Providing additional information to the user about the visualization
st.write("""
The histogram above represents the distribution of the selected categorical variable in the dataset. 
Adjust the bin width in the sidebar to change the granularity of the displayed data.
""")
# Correlation heatmap
# Displaying a subheader for the correlation heatmap section
st.subheader("Correlation Heatmap")

# Creating a figure for plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Using seaborn to plot a heatmap of correlations between numerical variables in the dataset
# Annotating each cell with the respective correlation coefficient, formatted to 2 decimal places
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)

# Adding a title to the plot
plt.title('Correlation Heatmap of Numeric Variables')

# Displaying the heatmap in the Streamlit app
st.pyplot(fig)

# Providing descriptive text to give context to the heatmap
st.write("""
The heatmap visualizes the correlation between different numerical variables in the dataset. 
Positive correlations are displayed in warm colors, while negative correlations are in cool colors. 
Values close to 1 or -1 represent strong positive or negative correlations, respectively.
""")

# Placeholder for the correlation matrix and type matrix
# Initializing matrices to store the correlation coefficients and types
corr_matrix = pd.DataFrame(np.zeros((len(data.columns), len(data.columns))), columns=data.columns, index=data.columns)
corr_type_matrix = corr_matrix.copy().astype(str)  

# Filling the correlation matrices
# Looping through each pair of variables to calculate correlations
for var1 in data.columns:
    for var2 in data.columns:
        # `calculate_correlations` assumed to be a user-defined function that returns the correlation and its type
        corr, corr_type = calculate_correlations(var1, var2, data, binary_vars, numeric_vars, multi_cat_vars)
        corr_matrix.loc[var1, var2] = corr  # Storing the correlation coefficient
        corr_type_matrix.loc[var1, var2] = corr_type  # Storing the correlation type

# Streamlit Display
# Displaying titles and introduction for the correlation analysis in the Streamlit app
st.title('Heart Disease Exploratory Analysis')
st.write('Correlation heatmaps representing relationships and correlation types between variables.')

# Plotting the Correlation Values
# Creating a figure to plot the correlation coefficients
fig, ax = plt.subplots(figsize=(12, 10))  

# Plotting the heatmap of correlation coefficients
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})

# Adding a title to the plot
plt.title("Correlation Coefficients")

# Displaying the plot in the Streamlit app
st.pyplot(fig)

# Providing explanatory text for the above heatmap
st.write('The above heatmap shows the correlation coefficients between variables using different correlation methods.')

# Plotting the Correlation Types
# Creating a figure to plot the correlation types
fig, ax = plt.subplots(figsize=(12, 10))

# Plotting a heatmap to visually display correlation types. Actual heatmap values are zero, annotation is used to display correlation types.
sns.heatmap(pd.DataFrame(np.zeros(corr_type_matrix.shape), columns=data.columns, index=data.columns), annot=corr_type_matrix, fmt="", cmap="Blues", cbar=False, ax=ax)

# Adding a title to the plot
plt.title("Correlation Types")

# Displaying the plot in the Streamlit app
st.pyplot(fig)

# Providing a legend for correlation types
st.write('Correlation Types: S = Spearman, P = Pearson, C = Chi-square')



# Scatter plots
st.sidebar.header('Scatter Plot Parameters')

# Selectboxes for choosing variables for scatter plot with default values set
x_axis = st.sidebar.selectbox('Select variable for x-axis', numeric_vars, index=0)
y_axis = st.sidebar.selectbox('Select variable for y-axis', numeric_vars, index=3)
color_var = st.sidebar.selectbox('Select variable for color', multi_cat_vars, index=multi_cat_vars.index('art_blocks'))
shape_var = st.sidebar.selectbox('Select variable for shape', binary_vars, index=binary_vars.index('is_male'))

# Filter warnings from Seaborn
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create scatter plot
st.subheader(f'Scatter plot of {y_axis} vs {x_axis}')
plt.figure(figsize=(12, 8))

# Using seaborn to create a scatterplot with hue (color) and style (shape) defined by categorical variables
sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=color_var, style=shape_var, palette="deep")

# Customize plot features
plt.title(f'{y_axis} vs {x_axis} colored by {color_var} and shaped by {shape_var}')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend(title=f'{color_var}/{shape_var}', bbox_to_anchor=(1, 1), loc='upper left')

# Display plot in Streamlit
st.pyplot()


