# streamlit project
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go


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
    'art_blocks': "Diagnosis of heart disease (Number of >50% diameter arteries: 0-3)."
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
st.markdown("""### Comments on Dataset
This dataset, sourced from the UCI ML Repo, contains 303 observations across 14 features. 
One feature, `num_fluoro`, was removed due to redundancy and missing data. The primary aim is to investigate predictors of arterial blockages, utilizing the remaining features.
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




# ... [Previous code for setup]

# 1. Compute z-scores for numeric variables
for var in numeric_vars:
    data[f"{var}_zscore"] = (data[var] - np.mean(data[var])) / np.std(data[var])

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

    # Display the figure with increased size
    st.plotly_chart(fig, use_container_width=True, height=800)
else:
    st.write("Please select at least one group of variables to display the graph.")

# ... [rest of the code]






# Define the color scale based on the unique values in art_blocks
unique_blocks = sorted(data['art_blocks'].unique())
color_scale = px.colors.qualitative.Plotly[:len(unique_blocks)]

# Map each unique block to its corresponding color
color_map = dict(zip(unique_blocks, color_scale))

st.header("Violin Plots")
st.markdown("These plots provide insights into the distribution of different variables across arterial blockage categories.")
# Violin Plots
for group in [group1, group2, group3, group4]:
    for feature in group:
        fig = px.violin(data, y=feature, x="art_blocks", color="art_blocks", 
                        box=True, points="all", color_discrete_map=color_map)
        st.plotly_chart(fig)


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

# Initialize matrix for correlation values
correlation_value_df = pd.DataFrame(index=all_vars, columns=all_vars)

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

st.plotly_chart(fig, use_container_width=True)

# Display the variables that used Spearman or Chi-square
st.write("Variables that used Spearman coefficient:", ', '.join(sorted(spearman_variables)))
st.write("Variables that used Chi-square coefficient:", ', '.join(sorted(chi_square_variables)))



# Selection widgets
x_var = st.selectbox('Select X variable:', options=multi_cat_vars + numeric_vars)
y_var = st.selectbox('Select Y variable:', options=multi_cat_vars + numeric_vars)
color_var = st.selectbox('Select variable for color:', options=cat_vars)
shape_var = st.selectbox('Select variable for shape:', options=binary_vars)

# Plot scatter plot
fig = px.scatter(data, x=x_var, y=y_var, color=color_var, symbol=shape_var, 
                 title=f'Scatter plot of {x_var} vs. {y_var}',
                 labels={color_var: f'Color by {color_var}', shape_var: f'Shape by {shape_var}'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)

st.plotly_chart(fig)


