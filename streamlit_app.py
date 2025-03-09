import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

# Set Streamlit layout to wide and add a title to the page
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Apply custom CSS for a unique theme and dark background
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #333, #111);
        color: white;
    }
    .sidebar .sidebar-content {
        background: #333;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Generate a synthetic dataset for demonstration
np.random.seed(2)
feature_names = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
X = np.random.rand(1000, 9)  # 1000 samples, 9 features
y = np.random.randint(0, 2, 1000)  # Binary target variable
X = pd.DataFrame(X, columns=feature_names)

# Mapping numerical data to categorical columns for demonstration purposes
X['Sex'] = np.random.choice(['Male', 'Female'], size=1000)
X['ChestPainType'] = np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], size=1000)
X['ExerciseAngina'] = np.random.choice(['N', 'Y'], size=1000)
X['ST_Slope'] = np.random.choice(['Up', 'Flat', 'Down'], size=1000)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define column transformer for preprocessing
categorical_features = ['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
numerical_features = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Preprocessing pipeline with OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)  # Use drop='first' to handle multicollinearity
    ])

# Define function to create model pipelines
def create_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Initialize the classifiers
lr_model = create_pipeline(LogisticRegression(random_state=2))
nb_model = create_pipeline(BernoulliNB())
lgb_model = create_pipeline(lgb.LGBMClassifier(random_state=2))
cb_model = create_pipeline(CatBoostClassifier(verbose=0, random_state=2))
svm_model = create_pipeline(SVC(probability=True, random_state=2))
rf_model = create_pipeline(RandomForestClassifier(random_state=2))
mlp_model = create_pipeline(MLPClassifier(random_state=2))

# Define a dictionary of models and their respective classifiers
models_and_classifiers = {
    'Logistic Regression': lr_model,
    'Bernoulli Naive Bayes': nb_model,
    'LightGBM': lgb_model,
    'CatBoost': cb_model,
    'Support Vector Machine': svm_model,
    'Random Forest': rf_model,
    'MLP Classifier': mlp_model
}

# Sidebar for model selection and input features
with st.sidebar:
    st.title("Model Selection")
    selected_models = st.multiselect(
        "Select Models to Test",
        options=list(models_and_classifiers.keys()),
        default=['Logistic Regression']
    )

    st.markdown("### Input Features")
    feature_inputs = {}

    for feature in feature_names:
        if feature in categorical_features:
            if feature == 'Sex':
                feature_inputs[feature] = st.selectbox(f"{feature}:", ["Male", "Female"])
            elif feature == 'ChestPainType':
                feature_inputs[feature] = st.selectbox(f"{feature}:", ["ATA", "NAP", "ASY", "TA"])
            elif feature == 'ExerciseAngina':
                feature_inputs[feature] = st.selectbox(f"{feature}:", ["N", "Y"])
            elif feature == 'ST_Slope':
                feature_inputs[feature] = st.selectbox(f"{feature}:", ["Up", "Flat", "Down"])
        else:
            feature_inputs[feature] = st.number_input(f"{feature}:", min_value=0.0, step=1.0)

# Main content with model testing and information
tabs = st.tabs(["Test Models", "Model Information", "Visualizations", "Predict Heart Disease"])

with tabs[0]:
    st.header("Test Models")
    if st.button("Run Model Tests"):
        with st.spinner("Testing models..."):
            for model_name in selected_models:
                model = models_and_classifiers[model_name]
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.subheader(f"Results for {model_name}")
                    st.text(f"Classification Report for {model_name}:")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_estimator(model.named_steps['classifier'], X_test, y_test, ax=ax)
                    st.pyplot(fig)
                    
                    # Plot AUC-ROC curve
                    y_proba = model.named_steps['classifier'].predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_roc = roc_auc_score(y_test, y_proba)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f'{model_name} (area = {auc_roc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                except NotFittedError:
                    st.error(f"The model {model_name} has not been fitted yet. Please run the model tests first.")
                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")

with tabs[1]:
    st.header("Model Information")
    st.write("Select models and input features to test models for heart disease predictions.")

with tabs[2]:
    st.header("Visualizations")
    st.write("Check out visualizations used in this study.")

    # Define the folder where the images are stored
    image_folder = "heart_infections"
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) < 26:
            st.warning("Not enough images found in the 'heart_infections' folder. Expected 26, found fewer.")

        # Display the first 26 images in the folder
        for image_file in image_files[:26]:
            st.image(os.path.join(image_folder, image_file), use_column_width=True, caption=image_file)  # Display image with caption
    else:
        st.warning(f"The folder '{image_folder}' does not exist.")

with tabs[3]:
    st.header("Predict Heart Disease")
    st.write("Input the features to predict if there is heart disease or not.")

    # Convert user inputs into the correct format
    user_input = {}
    for feature in feature_names:
        user_input[feature] = feature_inputs[feature]

    user_input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        for model_name in selected_models:
            model = models_and_classifiers[model_name]
            try:
                # Ensure model is fitted with training data
                model.fit(X_train, y_train)
                user_input_transformed = model.named_steps['preprocessor'].transform(user_input_df)
                prediction = model.named_steps['classifier'].predict(user_input_transformed)
                probability = model.named_steps['classifier'].predict_proba(user_input_transformed)[:, 1]
                st.subheader(f"Prediction using {model_name}:")
                st.write("Heart Disease" if prediction[0] == 1 else "No Heart Disease")
                st.write(f"Probability: {probability[0]:.2f}")
            except NotFittedError:
                st.error(f"The model {model_name} has not been fitted yet. Please run the model tests first.")
            except Exception as e:
                st.error(f"Error with model {model_name}: {e}")

# Footer with developer information
st.markdown(
    """
    Developed by: Ayodele Kolawole |
    """
)
