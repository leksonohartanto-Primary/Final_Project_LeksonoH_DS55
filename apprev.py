import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION AND MODEL LOADING
# ============================================================================

st.set_page_config(layout="wide", page_title="Income Prediction Dashboard")

# Load the trained model and scaler
try:
    # XGBoost model with hyperparameter tuning is the final chosen model
    model = joblib.load('xgb_tune_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # Load the pre-calculated selected correlation matrix and features for heatmap
    selected_correlation_matrix = joblib.load('selected_correlation_matrix_updated.joblib') # UPDATED FILENAME
    selected_features = joblib.load('selected_features_updated.joblib') # UPDATED FILENAME
    # Load the pre-calculated feature correlations for the bar chart
    # Assuming feature_correlations.joblib still contains the aggregated correlations
    # based on the original full X_encoded features, as before.
    # If this also needs to be updated based on X_train_selected, it would need another .joblib file.
    # For now, keeping the original logic for top 10 features as it was already based on grouped features.
    feature_correlations = joblib.load('feature_correlations.joblib')
except FileNotFoundError:
    st.error("Model, scaler, correlation data or feature list file not found. Please ensure all required .joblib files are in the same directory.")
    st.stop()

# Feature list - MUST EXACTLY MATCH the order of X.columns from the training notebook
# These are the 49 selected features after correlation-based feature selection
model_columns = [
    'age', 'hours_per_week', 'sex', 'workclass_Federal-gov', 'workclass_Local-gov',
    'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
    'workclass_Unknown', 'education_10th', 'education_11th', 'education_12th',
    'education_1st-4th', 'education_5th-6th', 'education_7th-8th', 'education_9th',
    'education_Bachelors', 'education_Doctorate', 'education_HS-grad', 'education_Masters',
    'education_Prof-school', 'education_Some-college', 'marital_status_married',
    'marital_status_never married', 'marital_status_prev married', 'occupation_Adm-clerical',
    'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
    'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv',
    'occupation_Prof-specialty', 'occupation_Unknown', 'relationship_Husband',
    'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child',
    'relationship_Unmarried', 'relationship_Wife', 'race_Black', 'race_Other',
    'race_White', 'native_country_Other', 'native_country_United-States',
    'has_capital_gain_0', 'has_capital_gain_1', 'has_capital_loss_0', 'has_capital_loss_1'
] # UPDATED LIST

# Numerical columns that were scaled during training
numerical_cols_for_scaling = ['age', 'hours_per_week']

# ============================================================================
# FEATURE IMPORTANCE CALCULATION AND SORTING
# ============================================================================

def get_top_10_features():
    """Extract and return top 10 most important features (correlations) from the pre-calculated data.
    """
    if 'feature_correlations' not in st.session_state:
        st.warning("Feature correlation data not loaded.")
        return None

    try:
        # Directly use the loaded and sorted correlation series
        top_10_features = list(st.session_state.feature_correlations.head(10).items()) # Convert to list
        return top_10_features
    except Exception as e:
        st.warning(f"Error processing feature correlations: {e}")
        return None

# Store feature_correlations in session_state to avoid reloading on every rerun
if 'feature_correlations' not in st.session_state:
    st.session_state.feature_correlations = feature_correlations

# ============================================================================
# PAGE HEADER AND INTRODUCTION
# ============================================================================

# Header with image
top_left_col, top_right_col = st.columns([4, 1])
with top_right_col:
    try:
        st.image('image1.png', width=180)
    except:
        pass

st.title("📈 Income Prediction Dashboard")
st.markdown("""
**Welcome to the Income Prediction Dashboard!**

This project created by Dataverse Team (Digital Skola Data Science Bootcamp Batch 55)

Provide an individual's characteristics below and this application will predict whether their annual income is **>50K** or **≤50K**.

*Built with XGBoost model trained on the Adult Census Income dataset with hyperparameter tuning using RandomizedSearchCV.*
""")
st.markdown('---')



# ============================================================================
# DATA PREPROCESSING PIPELINE INFORMATION
# ============================================================================

with st.expander("ℹ️ Data Preprocessing Pipeline: Strategi Preprocessing Data dan Pelatihan Model"):
    st.markdown("""
    #### Data Preprocessing Steps Applied in Detail:

    Seluruh rangkaian proses ini dirancang untuk menjaga kualitas data, memastikan representasi fitur yang optimal, serta menghasilkan model yang tangguh. Berikut adalah detail langkah-langkahnya:


    1.  **Data Cleaning and Initial Handling:**
        -   **Data Loading & Concatenation:** adult.csv (data training) dan adult.test.csv (data testing) dimuat secara terpisah. Seluruh langkah preprocessing berikutnya (seperti penanganan missing values dan encoding) ditentukan hanya berdasarkan data training, lalu diterapkan ke kedua dataset guna menghindari data leakage.
        -   **Whitespace Removal:** Spasi di awal dan akhir pada semua kolom bertipe object/string dihapus secara mandiri, baik untuk data training maupun testing.
        -   **Missing Value Identification & Imputation:**
            - Karakter ? (yang menandakan data hilang) diganti menjadi NaN di kedua dataset.
            - Untuk baris dengan workclass berisi 'Never-worked' namun occupation kosong, bagian occupation diisi dengan 'No-occupation' pada kedua dataset.
            - Nilai yang hilang pada native_country diisi menggunakan mode (nilai yang paling sering muncul) dari native_country yang dihitung hanya dari data training. Nilai mode tersebut ('United-States') kemudian digunakan untuk mengisi data training maupun testing.
            - Sisa nilai NaN pada workclass dan occupation diisi dengan label 'Unknown' di kedua dataset.
        -   **Target Variable Standardization (income):** Label pada kolom income diseragamkan (misalnya, dari '>50K.' menjadi '>50K'), lalu diubah ke dalam format biner (0 untuk '≤50K' dan 1 untuk '>50K') secara mandiri di kedua dataset.
        -   **Chi-square Test untuk Sex dan Income:** Chi-square test dijalankan pada train_data untuk melihat hubungan antara 'sex' dan 'income'. Hasil p-value menunjukkan angka 0.0000, yang berarti terdapat hubungan yang signifikan antara jenis kelamin dan tingkat pendapatan, sehingga variabel ini layak digunakan sebagai predictive feature.

    2.  **Feature Engineering and Transformation:**
        -   **Redundant Feature Removal:** Variabel education_num dihapus dari kedua dataset karena informasinya sudah terwakili oleh variabel education.
        -   **Categorical Feature Grouping:**
            -   **Marital Status:** Kategori dikelompokkan menjadi tiga kelompok besar: 'married', 'never married', dan 'prev married'. Aturan pengelompokan ini disusun berdasarkan data training dan diterapkan secara konsisten pada kedua dataset.
            -   **Native Country:** Semua negara selain 'United-States' dikelompokkan ke dalam kategori 'Other' berdasarkan pengamatan pada data training dan diterapkan ke kedua dataset.
        -   **Binary Capital Features:** Variabel capital_gain dan capital_loss diubah menjadi fitur biner (has_capital_gain, has_capital_loss: 1 jika >0, 0 jika =0) pada kedua dataset, lalu kolom aslinya dihapus.
        -   **Encoding Categorical Variables:**
            -   Fitur sex diproses menggunakan Label Encoded (Male=1, Female=0) dengan cara melakukan fitting LabelEncoder hanya pada data training, yang kemudian digunakan untuk mentransformasi data training maupun testing.
            -   Semua fitur kategori nominal lainnya (workclass, education, occupation, marital_status, relationship, race, native_country, has_capital_gain, has_capital_loss) diproses menggunakan One-Hot Encoded. Proses fitting OneHotEncoder dilakukan hanya pada data training sebelum digunakan untuk mentransformasi kedua dataset.
        -   **Feature Consolidation:** Seluruh fitur yang telah diproses (baik numerik, label-encoded, maupun one-hot-encoded) digabungkan untuk membentuk DataFrame X_train dan X_test.

    3.  **Feature Selection and Scaling:**
        -   **Correlation-Based Filtering:** Fitur yang memiliki nilai koefisien korelasi absolut kurang dari 0.03 terhadap variabel target income diidentifikasi hanya berdasarkan data training, lalu dihapus dari X_train maupun X_test. Langkah ini berhasil memangkas jumlah fitur dari 65 menjadi 49, sehingga efisiensi model meningkat dan gangguan (noise) pada data berkurang.
        -   **Feature Scaling:** Fitur numerik (age, hours_per_week) diseragamkan skalanya menggunakan StandardScaler. Proses fitting pada StandardScaler dilakukan hanya pada kolom numerik di data training (X_train_selected), yang kemudian digunakan untuk mentransformasi data training dan testing (X_train_scaled, X_test_scaled).

    4.  **Handling Class Imbalance with Synthetic Minority Over-sampling Technique (SMOTE):**
        -   Data X_train_scaled dan y_train yang sudah diproses kemudian dipecah menjadi dua bagian: set data untuk pengembangan latihan (x_dev_train, y_dev_train) dan set data untuk pengembangan pengujian (x_dev_test, y_dev_test).
        -   Setelah itu, SMOTE diterapkan hanya pada set data pengembangan latihan (x_dev_train, y_dev_train) guna menyeimbangkan distribusi kelas (dari yang tadinya sekitar 76:24 menjadi 50:50). Strategi ini dilakukan untuk memastikan tidak ada data sintetis yang masuk ke dalam set validasi maupun pengujian akhir, sehingga data leakage bisa dihindari dan evaluasi model menjadi jauh lebih realistis.

    5.  **Model Training and Selection:**
        -   **Initial Model Evaluation:** Berbagai algoritma klasifikasi (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVC, KNN) dievaluasi menggunakan 5-fold cross-validation pada data pengembangan latihan yang sudah diterapkan SMOTE. Metrik performa yang dipantau mencakup ROC AUC dan Accuracy.
        -   **Hyperparameter Optimization:** Model XGBoost dan Random Forest dioptimalkan lebih lanjut menggunakan RandomizedSearchCV pada data pengembangan latihan hasil SMOTE untuk menemukan parameter terbaik.
        -   **Final Model Selection:** XGBoost Classifier dengan parameter yang telah disetel menunjukkan performa yang paling unggul, sehingga terpilih sebagai model akhir.
        -   **Model Persistence:** Model XGBoost Classifier akhir yang sudah disetel (xgb_tune_model.joblib) beserta StandardScaler yang telah di-fit (scaler.joblib) disimpan menggunakan joblib untuk kebutuhan deployment.

    """
    )
st.markdown('---')

# ============================================================================
# USER INPUT FORM
# ============================================================================

st.header("👤 Individual Characteristics")
st.markdown("Adjust the values and select the options below to describe the individual:")

def user_input_features():
    """Create user input form with widgets for all features"""
    user_inputs = {}

    # Widget definitions mapping original feature names to widget creators
    widget_definitions = {
        'age': lambda: st.slider('Age', 17, 90, 35, help="Person's age in years"),
        'workclass': lambda: st.selectbox('Workclass',
            ['Private', 'Self-emp-not-inc', 'Local-gov', 'Unknown', 'State-gov',
             'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked'],
            help="Type of employment"),
        'education': lambda: st.selectbox('Education Level',
            ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc',
             '11th', 'Assoc-acdm', '10th', '7th-8th', '9th', 'Prof-school',
             '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'],
            help="Highest level of education completed"),
        'marital_status': lambda: st.selectbox('Marital Status',
            ['married', 'never married', 'prev married'], # Updated options
            help="Current marital status"),
        'occupation': lambda: st.selectbox('Occupation',
            ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical',
             'Sales', 'Other-service', 'Machine-op-inspct', 'Unknown',
             'Transport-moving', 'Handlers-cleaners', 'Farming-fishing',
             'Tech-support', 'Protective-serv', 'Priv-house-serv', 'No-occupation'],
            help="Type of occupation"),
        'relationship': lambda: st.selectbox('Relationship',
            ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'],
            help="Relationship to household head"),
        'race': lambda: st.selectbox('Race',
            ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            help="Race/ethnicity"),
        'sex': lambda: st.selectbox('Gender',
            ['Male', 'Female'],
            help="Gender"),
        'capital_gain': lambda: st.selectbox('Has Capital Gain?',
            ['No', 'Yes'],
            help="Whether the person has capital gains"),
        'capital_loss': lambda: st.selectbox('Has Capital Loss?',
            ['No', 'Yes'],
            help="Whether the person has capital losses"),
        'hours_per_week': lambda: st.slider('Hours per Week', 1, 99, 40, help="Number of hours worked per week"),
        'native_country': lambda: st.selectbox('Native Country',
            ['United-States', 'Other'],
            help="Country of origin")
    }

    # Organize inputs in columns
    num_cols_per_row = 3
    cols = st.columns(num_cols_per_row)
    col_index = 0

    # Use importance-based ordering if available, otherwise default order
    feature_order = [
        'age', 'hours_per_week', 'education', 'marital_status', 'occupation',
        'relationship', 'workclass', 'race', 'sex', 'capital_gain',
        'capital_loss', 'native_country'
    ]

    for feature_name in feature_order:
        if feature_name in widget_definitions:
            with cols[col_index]:
                user_inputs[feature_name] = widget_definitions[feature_name]()
            col_index = (col_index + 1) % num_cols_per_row
            if col_index == 0:
                cols = st.columns(num_cols_per_row)

    return pd.DataFrame(user_inputs, index=[0])

input_df_raw = user_input_features()

# ============================================================================
# INPUT PREPROCESSING
# ============================================================================

def preprocess_input_for_model(df_raw):
    """Preprocess user input to match the model's training format"""

    temp_df = df_raw.copy()

    # Convert numerical columns to float
    temp_df['age'] = temp_df['age'].astype(float)
    temp_df['hours_per_week'] = temp_df['hours_per_week'].astype(float)

    # Handle 'sex' (Label Encoding in notebook: Female=0, Male=1). Matches notebook now.
    temp_df['sex'] = 1 if temp_df['sex'].iloc[0] == 'Male' else 0

    # Create binary capital features as string '1'/'0' for OHE (as done in notebook)
    temp_df['has_capital_gain'] = '1' if temp_df['capital_gain'].iloc[0] == 'Yes' else '0'
    temp_df['has_capital_loss'] = '1' if temp_df['capital_loss'].iloc[0] == 'Yes' else '0'

    # Remove original capital_gain and capital_loss columns as they are replaced by has_capital_gain/loss
    temp_df = temp_df.drop(columns=['capital_gain', 'capital_loss'])

    # Define columns to be one-hot encoded (excluding 'sex' which is already handled)
    # The 'has_capital_gain' and 'has_capital_loss' are treated as categorical for OHE, as their values are '0' or '1' strings.
    ohe_input_cols = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'native_country',
        'has_capital_gain', 'has_capital_loss' # These will be one-hot encoded to _0 and _1 variants
    ]

    # Apply one-hot encoding without dropping the first category
    processed_df = pd.get_dummies(temp_df, columns=ohe_input_cols, drop_first=False)

    # Convert boolean columns to integers (pd.get_dummies might return bool)
    for col in processed_df.columns:
        if processed_df[col].dtype == 'bool':
            processed_df[col] = processed_df[col].astype(int)

    # Reindex to match model columns exactly, filling missing with 0
    final_df = processed_df.reindex(columns=model_columns, fill_value=0)

    # Ensure correct data types (esp. for scaled numerical columns)
    for col in model_columns:
        if col in numerical_cols_for_scaling:
            final_df[col] = final_df[col].astype(float)
        else: # All other columns are expected to be int (binary or label encoded)
            final_df[col] = final_df[col].astype(int)


    # Apply scaling to numerical columns
    final_df[numerical_cols_for_scaling] = scaler.transform(final_df[numerical_cols_for_scaling])

    return final_df

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================

st.markdown('---')

# Centered prediction button with custom styling
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 15px 32px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    predict_clicked = st.button(
        '🚀 Predict Income',
        help="Click to get the income prediction based on the entered characteristics"
    )

if predict_clicked:
    # Preprocess and make prediction
    processed_input = preprocess_input_for_model(input_df_raw)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    # Display results
    st.subheader('🎯 Prediction Results')
    st.markdown('---')

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.write("### Predicted Income Level:")
        if prediction[0] == 1:
            st.success("### ✅ Income **>50K** (High Income)", icon="🎉")
        else:
            st.info("### ℹ️ Income **≤50K** (Low Income)", icon="📉")

    with result_col2:
        st.write("### Prediction Confidence:")
        confidence_low = prediction_proba[0][0]
        confidence_high = prediction_proba[0][1]

    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(
            label="≤50K Probability",
            value=f"{confidence_low:.1%}",
            delta=None
            )
    with col_metric2:
        st.metric(
            label=" >50K Probability",
            value=f"{confidence_high:.1%}",
            delta=None
            )

    # Confidence gauge chart
    st.markdown('---')
    st.write("### Confidence Gauge:")

    fig_gauge = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=confidence_high * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence for >50K Income (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    )])
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('---')
else:
    st.info("👆 Adjust the features above and click '🚀 Predict Income' to see the results!")

# ============================================================================
# FEATURE IMPORTANCE DISPLAY (TOP 10 BAR CHART)
# ============================================================================

st.subheader("📊 Top 10 Most Important Features (Correlation with Income)")
st.markdown("""
Based on the **correlation with the target income variable**, these are the features that show the strongest relationship with the individual's income level.
""")

top_10_features = get_top_10_features()

if top_10_features:
    # Create DataFrame for visualization
    top_10_df = pd.DataFrame(list(top_10_features), columns=['Feature', 'Correlation'])

    # Create interactive bar chart
    fig_importance = px.bar(
        top_10_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Top 10 Features by Absolute Correlation with Income',
        labels={'Correlation': 'Absolute Correlation Coefficient', 'Feature': 'Feature Name'},
        height=500,
        color='Correlation',
        color_continuous_scale='RdYlGn'
    )

    fig_importance.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest',
        margin=dict(l=200)
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # Display top 5 features as metrics
    st.subheader("🎯 Top 5 Most Influential Features Metrics")
    cols = st.columns(5)
    for idx, (feature, correlation) in enumerate(top_10_features[:5]):
        with cols[idx]:
            st.metric(
                label=f"#{idx + 1}",
                value=feature,
                delta=f"{correlation:.4f}"
            )
else:
    st.warning("Unable to display feature correlation data.")

st.markdown('---')

# ============================================================================
# FEATURE IMPORTANCE DISPLAY (TOP FEATURES HEATMAP)
# ============================================================================

st.subheader("📊 Correlation Heatmap of Top 15 Features (excluding Sex) with Income") # UPDATED TITLE
st.markdown("""
Based on their **correlation with the target income variable**, this heatmap displays the relationships between the 15 most influential features and income. Positive values indicate a direct relationship, negative values an inverse relationship.
""")

if 'selected_correlation_matrix' in locals() and 'selected_features' in locals():
    # Create a Plotly Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=selected_correlation_matrix.values,
        x=selected_correlation_matrix.columns,
        y=selected_correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1, # Ensure color scale covers full correlation range
        colorbar=dict(title='Correlation Coefficient')
    ))

    fig_heatmap.update_layout(
        title_text='Correlation Heatmap of Top 15 Features (excluding Sex) with Income',
        xaxis_nticks=36,
        yaxis_nticks=36,
        height=700, # Adjust height for better visibility
        width=800,  # Adjust width for better visibility
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_automargin=True,
        yaxis_automargin=True,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    # Add annotations for correlation values
    annotations = []
    for i, row in enumerate(selected_correlation_matrix.values):
        for j, value in enumerate(row):
            annotations.append(dict(x=selected_correlation_matrix.columns[j], y=selected_correlation_matrix.index[i],
                                    text=f'{value:.2f}',
                                    font=dict(color='white' if abs(value) > 0.6 else 'black' if abs(value) > 0.2 else 'gray'),
                                    showarrow=False))
    fig_heatmap.update_layout(annotations=annotations)

    st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.warning("Unable to display feature correlation heatmap.")

st.markdown('---')

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('---')
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Developed with Streamlit and XGBoost by Dataverse Team DS55</p>
    <p>Data Source: <a href='https://www.kaggle.com/datasets/uciml/adult-census-income'>UCI Machine Learning Repository - Adult Census Income</a></p>
    <p>Model: XGBoost with Hyperparameter Tuning (RandomizedSearchCV)</p>
</div>

""", unsafe_allow_html=True)
