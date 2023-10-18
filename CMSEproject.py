# streamlit project
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


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
    df = pd.read_csv('heart_disease_uci.csv', nrows=303)
    
    # Rename columns and deal with NaN values
    new_column_names = [
        'age', 'is_male', 'chest_pain', 'rest_bp', 'chol',
        'high_sugar', 'rest_ecg', 'max_hr', 'exercise_angina',
        'st_depression', 'st_slope', 'num_fluoro',
        'thalass_type', 'art_blocks'
    ]
    df.columns = new_column_names
    
    df = df.drop('num_fluoro', axis = 1)  # Duplicate of 'blockages'
    # Imputing missing values based on certain criteria
    df.at[87, 'thalass_type'] = 3.0  # Chosen based on mode for thalass when block == 0
    df.at[266, 'thalass_type'] = 7.0  # Chosen by comparing other patients with blockage & exercise angina and getting the mode
    
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
    if var1 in binary_vars and var2 in binary_vars:
        corr, _ = spearmanr(data[var1], data[var2])
        corr_type = "S"  # Spearman
    elif var1 in numeric_vars and var2 in numeric_vars:
        corr, _ = pearsonr(data[var1], data[var2])
        corr_type = "P"  # Pearson
    elif var1 in multi_cat_vars and var2 in multi_cat_vars:
        chi2, _, _, _ = chi2_contingency(pd.crosstab(data[var1], data[var2]))
        corr = np.sqrt(chi2 / (chi2 + data.shape[0]))
        corr_type = "C"  # Chi-square
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
    'age': "The patient's age.",
    'is_male': "Whether the patient is male (1) or female (0).",
    'chest_pain': """The kind of chest discomfort the patient feels:
        - Value 1: Typical heart-related pain
        - Value 2: Less common heart-related pain
        - Value 3: Pain not from the heart
        - Value 4: No discomfort at all""",
    'rest_bp': "Blood pressure when the patient first came in.",
    'chol': "Level of cholesterol in the blood.",
    'high_sugar': "If sugar level in the blood was high before breakfast (1 = yes; 0 = no).",
    'rest_ecg': """Heart's electrical activity at rest:
        - Value 0: Normal
        - Value 1: Slight abnormality
        - Value 2: Possible thickening of heart muscle""",
    'max_hr': "Highest heart rate during a physical test.",
    'exercise_angina': "Whether exercise caused heart-related discomfort (1 = yes; 0 = no).",
    'st_depression': "Change in the heart's activity during exercise compared to rest.",
    'st_slope': """How a specific part of the heart's activity changes with exercise:
        - Value 1: Increases normally
        - Value 2: Stays flat
        - Value 3: Decreases""",
    'num_fluoro': "Number of major blood vessels seen in a special X-ray (from 0-3).",
    'thalass_type': """Type of a blood condition:
        - Value 3: Normal
        - Value 6: A stable issue  (long term damage)
        - Value 7: A temporary issue (can heal)""",
    'art_blocks': "Number of major arteries blocked (from 0-3). More blockage can mean higher risk."
}
# Displaying Descriptive Information about Variables to the User
st.header("Variable Descriptions")
# Iterating through the predefined variable descriptions and displaying them
for var, desc in selected_vars_desc.items():
    st.subheader(var)
    st.write(desc)


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
    st.write(data.head(303))

# Providing Additional Comments and Context about the Dataset
st.markdown("""
### Comments on Dataset
This dataset, sourced from the UCI ML Repo, contains 303 observations across 14 features. 
One feature, `num_fluoro`, was removed due to redundancy and missing data. The primary aim is to investigate predictors of arterial blockages, utilizing the remaining features.

#### Notes on Data Cleaning:
- Some missing data was filled to make the dataset more comprehensive.
- For the row with index 87, the missing value in `thalass_type` was filled with 3.0. This was based on the most common value (mode) for `thalass_type` when there's no blockage (`art_block == 0`).
- For the row with index 266, the missing value in `thalass_type` was set to 7.0. This decision was made by comparing other patients with arterial blockage & exercise-induced discomfort. The mode of `thalass_type` in such cases was 7.0.
""")

# These are the variables for which violin plots will be generated
group1 = ['chol']
group2 = ['age']
group3 = ['st_depression']
group4 = ['rest_bp', 'max_hr']
bin_width = 30

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


# 1. Compute z-scores for numeric variables
for var in numeric_vars:
    data[f"{var}_zscore"] = (data[var] - np.mean(data[var])) / np.std(data[var])

# Provide information to users about variable selection and z-score normalization
st.info("You can choose which type of variables to display in the parallel coordinates graph. If you select numeric variables, you'll also have the option to display them using z-scores, which normalize the data around a mean of 0 and a standard deviation of 1.")

# Use checkboxes for selection of variable groups but style them like toggles
st.write("Select Variables for Display:")
include_binary_vars = st.checkbox('Include Binary Variables', value=True, key="binary_vars")
include_multi_cat_vars = st.checkbox('Include Multi-category Variables', value=True, key="multi_cat_vars")
include_numeric_vars = st.checkbox('Include Numeric Variables', value=True, key="numeric_vars")

# Initialize empty list for selected variables
selected_vars = []

# Append to the selected variables list based on checked boxes
if include_binary_vars:
    selected_vars += binary_vars
if include_multi_cat_vars:
    selected_vars += multi_cat_vars
if include_numeric_vars:
    selected_vars += numeric_vars

# If numeric_vars are selected, show the z-score checkbox
use_zscore = False
if include_numeric_vars:
    use_zscore = st.checkbox('Use Z-Scored Variables for Numeric Data?')

    if use_zscore:
        # Replace the selected numeric variables with their z-scored versions
        for var in numeric_vars:
            selected_vars[selected_vars.index(var)] = f"{var}_zscore"

# Create the parallel coordinates visualization only if at least one group of variables is selected
if selected_vars:
    fig = px.parallel_coordinates(data, 
                                  dimensions=selected_vars,
                                  color="art_blocks",
                                  labels={col: col for col in selected_vars},
                                  color_continuous_scale=px.colors.diverging.Tealrose)
    
    st.info("The parallel coordinates graph below is interactive. You can hover over individual lines to see detailed data points and use the toolbar on the top right to zoom, pan, and reset the view.")
    
    # Display the figure with increased size
    st.plotly_chart(fig, use_container_width=True, height=800)
else:
    st.write("Please select at least one group of variables to display the graph.")

# Define the color scale based on the unique values in art_blocks
unique_blocks = sorted(data['art_blocks'].unique())
color_scale = px.colors.qualitative.Plotly[:len(unique_blocks)]

# Map each unique block to its corresponding color
color_map = dict(zip(unique_blocks, color_scale))

st.header("Violin Plots")
st.info("Below, you'll find a series of violin plots. They show how different variables are distributed across arterial blockage categories. You can interact with each plot to understand the distribution better.")
# Violin Plots
for group in [group1, group2, group3, group4]:
    for feature in group:
        fig = px.violin(data, y=feature, x="art_blocks", color="art_blocks", 
                        box=True, points="all", color_discrete_map=color_map)
        st.plotly_chart(fig)

# Guide for the Histogram selection
st.markdown("""### Histograms
Below, you can customize a histogram. First, select the feature for the x-axis. Optionally, you can enable faceting to divide the histogram based on another feature.""")
# 1. Let users select the feature for the x-axis of the histogram
feature = st.selectbox("Select a feature for the histogram x-axis:", multi_cat_vars + binary_vars)

# 2. Checkbox to allow users to enable faceting
enable_faceting = st.checkbox('Enable faceting?')

if enable_faceting:
    # 3. Let users select the feature for faceting when checkbox is ticked
    # Defaulting to "art_blocks" by finding its index in the list
    default_index = multi_cat_vars.index('art_blocks')
    facet_feature = st.selectbox("Select a feature for faceting:", multi_cat_vars + binary_vars, index=default_index)
    
    # 4. Create the histogram faceted by the chosen feature
    fig = px.histogram(data, x=feature, nbins=int(data[feature].max() / bin_width), facet_col=facet_feature, color=facet_feature)
else:
    # 5. Create the histogram without faceting if checkbox is not ticked
    fig = px.histogram(data, x=feature, nbins=int(data[feature].max() / bin_width))

st.plotly_chart(fig)


all_vars = numeric_vars + binary_vars + multi_cat_vars


correlation_value_df = pd.DataFrame(index=all_vars, columns=all_vars)

# Correlation Matrix
st.markdown("""### Correlation Matrix
This heatmap displays correlations between different variables. Dark blue or red indicates strong correlation, while white suggests little to no correlation. Click on a heatmap cell to see detailed scatter plots for the corresponding variable pair.""")


spearman_variables = set()
chi_square_variables = set()

for i, var1 in enumerate(all_vars):
    for j, var2 in enumerate(all_vars):
        if i > j:  # Only lower triangle
            corr_val, corr_type = calculate_correlations(var1, var2, data, binary_vars, numeric_vars, multi_cat_vars)
            correlation_value_df.loc[var1, var2] = corr_val
            
            # Collect variables that used Spearman or Chi-square
            if corr_type == 'S':
                spearman_variables.add(var1)
                spearman_variables.add(var2)
            elif corr_type == 'C':
                chi_square_variables.add(var1)
                chi_square_variables.add(var2)
        else:
            correlation_value_df.loc[var1, var2] = np.nan

# Create heatmap for correlation values
fig = go.Figure()

heatmap = go.Heatmap(z=correlation_value_df.values, x=all_vars, y=all_vars,
                     colorscale="RdBu_r", zmin=-1, zmax=1)

fig.add_trace(heatmap)
fig.update_layout(title="Lower Triangle Correlation Matrix")

# Using st_plotly_events for capturing the click event on the heatmap
click_data = plotly_events(fig)

if click_data:
    point_data = click_data[0]
    x_var_init, y_var_init = point_data['x'], point_data['y']
else:
    x_var_init, y_var_init = multi_cat_vars[0], numeric_vars[0]  # Default initial values

combined_vars = multi_cat_vars + numeric_vars

# Scatter plot controls and description
st.markdown("""### Scatter Plot
After selecting variables in the correlation matrix, you can further customize the scatter plot below. Choose variables for the x and y axes, color, and shape. You also have options to normalize or standardize the numeric data. If desired, a regression line can be added to the scatter plot.""")


# Check if x_var_init and y_var_init exist in the combined list
x_var_index = combined_vars.index(x_var_init) if x_var_init in combined_vars else 0
y_var_index = combined_vars.index(y_var_init) if y_var_init in combined_vars else 0

# Selection widgets with updated index calculation
x_var = st.selectbox('Select X variable:', options=combined_vars, index=x_var_index)
y_var = st.selectbox('Select Y variable:', options=combined_vars, index=y_var_index)
color_var = st.selectbox('Select variable for color:', options=cat_vars)
shape_var = st.selectbox('Select variable for shape:', options=binary_vars)

# Dropdown for regression type
regression_type = st.selectbox(
    'Select regression type:',
    options=['None', 'ols', 'lowess']
)

# Toggle for z-scored/normed values
z_scored = st.button("Z-Score Numeric Variables")
normed = st.button("Normalize Numeric Variables")

if z_scored:
    for col in numeric_vars:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
elif normed:
    for col in numeric_vars:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# Scatter plot with regression line
fig = px.scatter(
    data, x=x_var, y=y_var, color=color_var, symbol=shape_var,
    trendline=regression_type if regression_type != 'None' else None,
    title=f'Scatter plot of {x_var} vs. {y_var}',
    labels={color_var: f'Color by {color_var}', shape_var: f'Shape by {shape_var}'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Adjust legend
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig)




