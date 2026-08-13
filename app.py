
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #173B67;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 17px;
    margin-bottom: 30px;
}

.info-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.result-low {
    background: #DCFCE7;
    border-left: 7px solid #22C55E;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-moderate {
    background: #FEF3C7;
    border-left: 7px solid #F59E0B;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-high {
    background: #FEE2E2;
    border-left: 7px solid #EF4444;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-title {
    font-size: 31px;
    font-weight: 800;
}

.small-note {
    color: #64748B;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 15px;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("student_stress_cleaned.csv")

    # Clean spaces from headings
    df.columns = df.columns.str.strip()

    return df


df = load_data()


# =========================================================
# TARGET
# =========================================================

TARGET = "Overall, how would you describe your current stress level?"

if TARGET not in df.columns:
    st.error(
        "Target column was not found in the dataset."
    )
    st.stop()


# =========================================================
# FEATURE / TARGET SPLIT
# =========================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =========================================================
# COLUMN TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =========================================================
# TARGET ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )
}


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    trained_models = {}
    results = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        prediction = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        trained_models[name] = pipeline
        results[name] = accuracy
        print(f"DEBUG: Model '{name}' calculated accuracy in app.py: {accuracy:.4f}") # Added debug print

    print(f"DEBUG: Final results dictionary from train_models in app.py: {results}") # Added debug print
    return trained_models, results


trained_models, results = train_models()
print(f"DEBUG: Results after calling train_models in app.py: {results}") # Added debug print

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Student Stress Level Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning based system for predicting
    student stress as Low, Moderate or High
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PROJECT SUMMARY
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Usable Records",
        len(df)
    )

with c2:
    st.metric(
        "Input Features",
        X.shape[1]
    )

with c3:
    st.metric(
        "ML Models",
        len(models)
    )

with c4:
    st.metric(
        "Stress Classes",
        len(label_encoder.classes_)
    )


st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔮 Stress Prediction",
    "📊 Model Comparison",
    "ℹ️ About Project"
])


# =========================================================
# TAB 1 — PREDICTION
# =========================================================

with tab1:

    st.subheader(
        "Enter Student Information"
    )

    st.caption(
        "Complete the fields below and press Predict Stress Level."
    )


    # Choose model
    selected_model_name = st.selectbox(
        "Prediction Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(
            "Decision Tree"
        )
    )


    with st.form("prediction_form"):

        user_input = {}

        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # CREATE INPUTS AUTOMATICALLY FROM DATASET
        # -------------------------------------------------

        for index, feature in enumerate(X.columns):

            container = (
                col1 if index % 2 == 0
                else col2
            )


            with container:

                # NUMERICAL FEATURE
                if feature in numeric_features:

                    minimum = float(
                        X[feature].min()
                    )

                    maximum = float(
                        X[feature].max()
                    )

                    default = float(
                        X[feature].median()
                    )


                    # Integer-like rating values
                    if (
                        X[feature]
                        .dropna()
                        .apply(float.is_integer)
                        .all()
                    ):

                        user_input[feature] = st.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(default)),
                            step=1
                        )

                    else:

                        user_input[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=0.01
                        )


                # CATEGORICAL FEATURE
                else:

                    options = (
                        X[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    user_input[feature] = st.selectbox(
                        feature,
                        options
                    )


        submitted = st.form_submit_button(
            "🔍 Predict Stress Level",
            use_container_width=True
        )


    # -------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------

    if submitted:

        input_df = pd.DataFrame(
            [user_input]
        )

        selected_model = trained_models[
            selected_model_name
        ]

        encoded_prediction = selected_model.predict(
            input_df
        )[0]

        prediction = label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]


        if prediction.lower() == "low":

            css_class = "result-low"
            icon = "🟢"

        elif prediction.lower() == "moderate":

            css_class = "result-moderate"
            icon = "🟡"

        else:

            css_class = "result-high"
            icon = "🔴"


        st.markdown(
            f"""
            <div class="{css_class}">
                <div>Predicted Stress Level</div>
                <div class="result-title">
                    {icon} {prediction.upper()}
                </div>
                <div>
                    Model used: {selected_model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.info(
            "This prediction is for academic demonstration "
            "and is not a medical diagnosis."
        )


# =========================================================
# TAB 2 — MODEL COMPARISON
# =========================================================

with tab2:

    st.subheader(
        "Machine Learning Model Comparison"
    )


    result_df = pd.DataFrame({

        "Model":
            list(results.keys()),

        "Accuracy (%)":
            [
                round(value * 100, 2)
                for value in results.values()
            ]
    })


    result_df = result_df.sort_values(
        "Accuracy (%)",
        ascending=False
    )


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Accuracy Visualization"
    )


    chart_df = (
        result_df
        .set_index("Model")
    )

    st.bar_chart(
        chart_df["Accuracy (%)"]
    )


    best_name = max(
        results,
        key=results.get
    )

    best_accuracy = results[
        best_name
    ] * 100


    st.success(
        f"Highest test accuracy: "
        f"{best_accuracy:.2f}% "
        f"({best_name})"
    )


    st.markdown(
        """
        **Models incorporated**

        - Logistic Regression
        - Decision Tree
        - Random Forest
        - K-Nearest Neighbors
        - Support Vector Machine
        """
    )


# =========================================================
# TAB 3 — ABOUT
# =========================================================

with tab3:

    st.subheader(
        "About the Project"
    )


    st.markdown(
        """
        ### Objective

        Predict a student's current stress level as:

        **Low • Moderate • High**

        using academic, lifestyle and
        self-reported personal factors.

        ### Workflow

        **Google Form Survey → Data Cleaning → EDA →
        Encoding & Scaling → Train/Test Split →
        Multiple ML Models → Evaluation → Web Prediction**

        ### Dataset

        The project uses an anonymous student survey dataset.

        ### Evaluation

        Models are evaluated using a held-out test set.

        ### Regularisation

        Logistic Regression uses regularisation by default,
        while class balancing is used where applicable.
        The notebook also compares L1 and L2 regularisation.

        ### Limitation

        This system is an academic ML project.
        It should not be interpreted as a clinical
        mental-health assessment.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <p style="text-align:center;color:#64748B">
    Student Stress Level Prediction • Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True
)
