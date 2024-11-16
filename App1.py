# App to predict the prices of diamonds using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt 

# Set up the app title and image
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', use_column_width = True)
st.write("Utilize our advanced Machine Learning application to predict fetal health classification")

# Reading the pickle file that I created before 
rf_pickle = open('rf_ml.pickle', 'rb') 
rf_clf = pickle.load(rf_pickle) 
rf_pickle.close()

dt_pickle = open('dt_ml.pickle', 'rb')
dt_clf = pickle.load(dt_pickle)
dt_pickle.close()

ab_pickle = open('ab_ml.pickle', 'rb')
ab_clf = pickle.load(ab_pickle)
ab_pickle.close()

sv_pickle = open('sv_ml.pickle', 'rb')
sv_clf = pickle.load(sv_pickle)
sv_pickle.close()

# Load the default dataset
original_df = pd.read_csv('fetal_health.csv')
# Remove output (species) and year columns from original data
original_df = original_df.drop(columns = ['fetal_health'])

# Sidebar for user input
with st.sidebar:
    st.header("Fetal Health Features Input")
    health_file = st.file_uploader("Upload your data", help = "File must be in CSV format")
    st.warning('Ensure your data strictly follows the format outlined below', icon="⚠️")
    st.dataframe(original_df.head())
    user_model = st.radio("Choose Model for Prediction",
                          ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"])
    st.info(f'You selected: {user_model}', icon="✅")

if user_model == "Random Forest":
    clf = rf_clf
elif user_model == "Decision Tree":
    clf = dt_clf
elif user_model == "AdaBoost":
    clf = ab_clf
else:
    clf = sv_clf

# Loading data
if health_file is not None:
    user_df = pd.read_csv(health_file)
else:
    st.info('Please upload data to proceed', icon="ℹ️")
    st.stop()

st.success('CSV file uploaded successfully', icon="✅")
st.header("Predicting Fetal Health Class Using")

# Dropping null values
user_df = user_df.dropna() 
original_df = original_df.dropna() 

# Ensure the order of columns in user data is in the same order as that of original data
user_df = user_df[original_df.columns]

# Concatenate two dataframes together along rows (axis = 0)
combined_df = pd.concat([original_df, user_df], axis = 0)

# Number of rows in original dataframe
original_rows = original_df.shape[0]

# Create dummies for the combined dataframe
combined_df_encoded = pd.get_dummies(combined_df)

# Split data into original and user dataframes using row index
original_df_encoded = combined_df_encoded[:original_rows]
user_df_encoded = combined_df_encoded[original_rows:]

# Predictions for user data
user_pred = clf.predict(user_df_encoded)

# Prediction probabilities
user_prob = clf.predict_proba(user_df_encoded)

# Adding predicted species to user dataframe
user_df['Predicted Fetal Health'] = user_pred
user_df['Prediction Probability (%)'] = (user_prob.max(axis=1)*100).round()

# Color cells
def cond_color(x):
    if x == "Normal":
        return "background-color: lime"
    elif x == "Suspect": 
        return "background-color: yellow"
    else:
        return "background-color: orange"

user_df = user_df.style.applymap(cond_color, subset=['Predicted Fetal Health'])

# Show the predicted species on the app
st.dataframe(user_df)

# Additional tabs for model performance

if user_model == "Random Forest":
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", 
                                "Classification Report", 
                                "Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_rf.svg')
        st.caption("Confusion Matrix of Predicted vs True output.")
    with tab2:
        st.write("### Classification Report")
        class_report = pd.read_csv('class_report_rf.csv')
        st.dataframe(class_report)
        st.caption("Precision, Recall, F1-Score, and Support for each health condition.")
    with tab3:
        st.write("### Feature Importance")
        st.image('feature_imp_rf.svg')
        st.caption("Relative importance of features in prediction.")
elif user_model == "Decision Tree":
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", 
                                "Classification Report", 
                                "Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_dt.svg')
        st.caption("Confusion Matrix of Predicted vs True output.")
    with tab2:
        st.write("### Classification Report")
        class_report = pd.read_csv('class_report_dt.csv')
        st.dataframe(class_report)
        st.caption("Precision, Recall, F1-Score, and Support for each health condition.")
    with tab3:
        st.write("### Feature Importance")
        st.image('feature_imp_dt.svg')
        st.caption("Relative importance of features in prediction.")
elif user_model == "AdaBoost":
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", 
                                "Classification Report", 
                                "Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_ab.svg')
        st.caption("Confusion Matrix of Predicted vs True output.")
    with tab2:
        st.write("### Classification Report")
        class_report = pd.read_csv('class_report_ab.csv')
        st.dataframe(class_report)
        st.caption("Precision, Recall, F1-Score, and Support for each health condition.")
    with tab3:
        st.write("### Feature Importance")
        st.image('feature_imp_ab.svg')
        st.caption("Relative importance of features in prediction.")
else:
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", 
                                "Classification Report", 
                                "Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image('confusion_mat_sv.svg')
        st.caption("Confusion Matrix of Predicted vs True output.")
    with tab2:
        st.write("### Classification Report")
        class_report = pd.read_csv('class_report_sv.csv')
        st.dataframe(class_report)
        st.caption("Precision, Recall, F1-Score, and Support for each health condition.")
    with tab3:
        st.write("### Feature Importance")
        st.image('feature_imp_sv.svg')
        st.caption("Relative importance of features in prediction.")



