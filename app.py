import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import difflib
import warnings

# ============================================================
# 1. SUPPRESS WARNINGS & CUSTOM CLASS IMPORTS
# ============================================================
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# 2. CUSTOM CLASSES (Required to load the CatBoost pipeline)
# ============================================================

class CorrelationVIFSelector(BaseEstimator, TransformerMixin):
    """Removes correlated & high-VIF features but PROTECTS specified indices."""
    def __init__(self, corr_threshold=0.9, vif_threshold=10, protect_indices=None):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.protect_indices = protect_indices if protect_indices is not None else []
        self.keep_indices_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features <= 1:
            self.keep_indices_ = np.arange(n_features)
            return self

        protected = set(self.protect_indices)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr[i, j]) > self.corr_threshold:
                    high_corr_pairs.append((i, j))
        mean_abs_corr = np.mean(np.abs(corr), axis=0)
        drop_set = set()
        for i, j in high_corr_pairs:
            if i in drop_set or j in drop_set:
                continue
            if i in protected and j in protected:
                continue
            if i in protected:
                drop_set.add(j)
            elif j in protected:
                drop_set.add(i)
            else:
                if mean_abs_corr[i] >= mean_abs_corr[j]:
                    drop_set.add(i)
                else:
                    drop_set.add(j)
        keep_indices = [i for i in range(n_features) if i not in drop_set]

        unprotected_remaining = [i for i in keep_indices if i not in protected]
        if len(unprotected_remaining) >= 2:
            X_rem = X[:, unprotected_remaining]
            stds = np.std(X_rem, axis=0)
            non_const = [idx for idx, s in enumerate(stds) if s > 1e-10]
            if len(non_const) >= 2:
                X_vif = X_rem[:, non_const]
                vif_data = np.zeros(len(non_const))
                for idx in range(len(non_const)):
                    try:
                        vif_data[idx] = variance_inflation_factor(X_vif, idx)
                    except Exception:
                        vif_data[idx] = 0.0
                vif_drop_rel = [non_const[i] for i, v in enumerate(vif_data) if v >= self.vif_threshold]
                vif_drop_abs = [unprotected_remaining[i] for i in vif_drop_rel]
                keep_indices = [i for i in keep_indices if i not in vif_drop_abs]

        self.keep_indices_ = np.array(sorted(set(keep_indices) | protected))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_indices_]


class RFSelectorKeepProtected(BaseEstimator, TransformerMixin):
    """RF-based embedded feature selection that ALWAYS keeps protected indices."""
    def __init__(self, n_estimators=100, random_state=42, n_jobs=1, threshold='median', keep_indices=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.threshold = threshold
        self.keep_indices = keep_indices if keep_indices is not None else []
        self.support_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rf_.fit(X, y)
        self.feature_importances_ = self.rf_.feature_importances_
        if self.threshold == 'median':
            thresh = np.median(self.feature_importances_)
        elif self.threshold == 'mean':
            thresh = np.mean(self.feature_importances_)
        else:
            thresh = float(self.threshold)
        keep_set = set(self.keep_indices)
        self.support_ = np.array([
            (i in keep_set) or (self.feature_importances_[i] >= thresh)
            for i in range(len(self.feature_importances_))
        ])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.support_]


class CleanCommunityTransformer(BaseEstimator, TransformerMixin):
    """Cleans community variables: strips '%', converts to numeric, handles missing."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].str.replace('%', '', regex=False).str.strip()
                    X[col] = pd.to_numeric(X[col], errors='coerce')
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)


HIDDEN_LAYER_OPTIONS = {
    "64": (64,),
    "128": (128,),
    "64_64": (64, 64),
    "128_64": (128, 64),
    "256_128": (256, 128),
    "128_64_32": (128, 64, 32),
}


class CatBoostSklearnWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=500, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3.0, border_count=128, random_strength=1.0,
                 bagging_temperature=1.0, min_data_in_leaf=1,
                 auto_class_weights="Balanced", verbose=0, random_state=42,
                 thread_count=2):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.min_data_in_leaf = min_data_in_leaf
        self.auto_class_weights = auto_class_weights
        self.verbose = verbose
        self.random_state = random_state
        self.thread_count = thread_count

    def _make_model(self):
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            min_data_in_leaf=self.min_data_in_leaf,
            auto_class_weights=self.auto_class_weights,
            verbose=self.verbose,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            allow_writing_files=False,
        )

    def fit(self, X, y):
        self.model_ = self._make_model()
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float)).flatten().astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))


class MLPSklearnWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes_code="64", activation="relu", solver="adam",
                 alpha=0.0001, learning_rate="constant", learning_rate_init=0.001,
                 batch_size=200, random_state=42, max_iter=500, early_stopping=True,
                 n_iter_no_change=15, validation_fraction=0.1):
        self.hidden_layer_sizes_code = hidden_layer_sizes_code
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction

    def _make_model(self):
        return MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_OPTIONS[self.hidden_layer_sizes_code],
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            random_state=self.random_state,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
        )

    def fit(self, X, y):
        self.model_ = self._make_model()
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y))
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))


# ============================================================
# 3. STREAMLIT UI STARTS HERE
# ============================================================

# Set page config FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Water Access Prediction - East Africa",
    page_icon="💧",
    layout="wide"
)

# Load model with no loading message
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("best_model.pkl")
    threshold = float(open("best_threshold.txt").read())
    return model, threshold

model, CATBOOST_THRESHOLD = load_model()

# ============================================================
# 4. PAGE CONTENT
# ============================================================
st.title("💧 Machine learning-based prediction of limited access to drinking water services among households using improved water source in East Africa: evidence from recent DHS data")
st.markdown("### CatBoost Model — East Africa Regional Prediction Tool")
st.markdown("---")

# ============================================================
# 5. SIDEBAR - MODEL INFORMATION
# ============================================================
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Algorithm:** CatBoost (Gradient Boosting)  
    **Region:** East Africa  
    **Outcome:** Limited vs Basic Water Access  
    **Features:** 38 household & community variables  
    """)
    
    st.metric("Classification Threshold (OOF)", f"{CATBOOST_THRESHOLD:.3f}")
    st.caption("Optimal threshold selected via out-of-fold CV")
    
    st.divider()
    st.markdown("**Performance Metrics (Test Set)**")
    st.markdown(f"""
    - **Accuracy:** 0.741 (0.736-0.746)
    - **Precision:** 0.799 (0.794-0.804)
    - **Recall:** 0.741 (0.736-0.746)
    - **F1-Score:** 0.759 (0.754-0.763)
    - **ROC-AUC:** 0.808 (0.802-0.813)
    - **AP-AUC:** 0.498 (0.485-0.512)
    """)
    
    st.divider()
    st.caption("🏆 CatBoost was selected as the best model among 9 algorithms")

# ============================================================
# 6. MAIN INPUT FORM
# ============================================================
st.header("📋 Enter Household Information")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Household Demographics")
    
    # Country
    country = st.selectbox(
        "Country",
        ["Burundi", "Ethiopia", "Kenya", "Madagascar", "Malawi", 
         "Mozambique", "Rwanda", "Tanzania", "Uganda", "Zambia", "Zimbabwe"]
    )
    
    # Type of place of residence
    residence = st.radio("Type of Place of Residence", ["rural", "urban"], horizontal=True)
    
    # Sex of head of household
    sex_head = st.radio("Sex of Household Head", ["female", "male"], horizontal=True)
    
    # Current marital status
    marital_status = st.selectbox(
        "Current Marital Status of Head",
        ["Never married", "Married", "Divorced", "Widowed"]
    )
    
    # Highest educational level
    education = st.selectbox(
        "Highest Educational Level Attained",
        ["No education", "Primary", "Secondary", "Higher"]
    )
    
    # Wealth index
    wealth = st.selectbox(
        "Wealth Index Combined",
        ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    )
    
    # Household size
    hh_size = st.radio("Number of Household Members", ["<4", ">=4"], horizontal=True)
    
    # Children under 5 group
    children_u5 = st.selectbox(
        "Children Under 5 Group",
        ["no child", "1-2", ">=3"]
    )
    
    # Head age group
    head_age = st.selectbox(
        "Household Head Age Group",
        ["<35", "35-60", ">60"]
    )

with col2:
    st.subheader("🏡 Household Facilities & Services")
    
    # Electricity
    electricity = st.radio("Has Electricity", ["no", "yes"], horizontal=True)
    
    # Water treatment
    water_treatment = st.radio("Household Water Treatment", ["no", "yes"], horizontal=True)
    
    # Media exposure
    media_exposure = st.radio("Household Media Exposure", ["No", "Yes"], horizontal=True)
    
    # Sanitation status
    sanitation = st.radio(
        "Sanitation Status",
        ["unimproved_sanitation", "improved_sanitation"],
        horizontal=True
    )
    
    # Handwashing place observed
    handwashing = st.radio("Handwashing Place Observed", ["no", "yes"], horizontal=True)
    
    # Share toilet
    share_toilet = st.radio("Share Toilet with Other Households", ["no", "yes"], horizontal=True)
    
    # Soap/detergent present
    soap = st.radio("Items Present: Soap or Detergent", ["no", "yes"], horizontal=True)
    
    # Housing material status
    housing_material = st.selectbox(
        "Housing Material Status",
        ["unimproved_housing_material", "improved_housing_material"]
    )
    
    # Person fetching water
    person_fetching = st.selectbox(
        "Person Fetching Water",
        ["Adult man", "Adult woman", "female_child_under_15", "male_child_under_15", "others"]
    )

# ============================================================
# 7. COMMUNITY LEVEL VARIABLES (Low / High select boxes)
# ============================================================
st.subheader("🌍 Community-Level Variables")
st.caption("Select 'High' if the community has a high percentage (>50%) of households with the facility, otherwise select 'Low'")

col3, col4 = st.columns(2)

with col3:
    community_electricity = st.selectbox("Community Level Electricity", ["Low", "High"])
    community_water_treatment = st.selectbox("Community Level Water Treatment", ["Low", "High"])
    community_media = st.selectbox("Community Level Media Exposure", ["Low", "High"])

with col4:
    community_poverty = st.selectbox("Community Level Poverty", ["Low", "High"])
    community_education = st.selectbox("Community Level Education", ["Low", "High"])
    community_toilet = st.selectbox("Community Level Toilet Facility", ["Low", "High"])

# ============================================================
# 8. PREDICTION FUNCTION (Passes raw inputs to the pipeline)
# ============================================================
def prepare_input_data():
    """Creates a DataFrame with raw feature names matching the model's training data."""
    
    # Helper to convert "Low"/"High" to numeric percentages (0 or 100)
    # The model was trained on percentages, so Low = 0%, High = 100%
    def map_community(value):
        return 0 if value == "Low" else 100
    
    # Build a dictionary with the exact column names from the training script
    input_dict = {
        'country': country,
        'type_of_place_of_residence': residence,
        'has_electricity': electricity,
        'sex_of_head_of_household': sex_head,
        'household_water_treatment': water_treatment,
        'shared_toilets_with_others': share_toilet,
        'handwashing_place_observed': handwashing,
        'number_of_households_members': hh_size,
        'items_present_soap_or_detergent': soap,
        'household_media_exposure': media_exposure,
        'sanitation_status': sanitation,
        'housing_material_status': housing_material,
        'person_fetching_water': person_fetching,
        'current_marital_status': marital_status,
        'children_under5_group': children_u5,
        'age_of_household_head': head_age,
        'highest_educational_level_attained': education,
        'wealth_index_combined': wealth,
        # --- Community variables mapped to numbers (0 or 100) ---
        'community_level_electricity': map_community(community_electricity),
        'community_level_water_treatment': map_community(community_water_treatment),
        'community_level_media_exposure': map_community(community_media),
        'community_level_poverty': map_community(community_poverty),
        'community_level_education': map_community(community_education),
        'community_level_toilet': map_community(community_toilet),
    }
    
    return pd.DataFrame([input_dict])

# ============================================================
# 9. PREDICTION BUTTON AND OUTPUT
# ============================================================
st.markdown("---")
if st.button("🔍 Predict Water Access Status", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing household characteristics with CatBoost..."):
        # Get raw input data
        input_df = prepare_input_data()
        
        # Get prediction probability (probability of Limited Access = class 1)
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Create metrics row
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Predicted Probability", f"{probability:.2%}")
        
        with col_result2:
            risk_level = "HIGH RISK" if probability >= CATBOOST_THRESHOLD else "LOW RISK"
            st.metric("Risk Level", risk_level)
        
        with col_result3:
            classification = "Limited Access" if probability >= CATBOOST_THRESHOLD else "Basic Access"
            st.metric("Water Access Classification", classification)
        
        # Show threshold information
        st.caption(f"*Classification based on optimal threshold: {CATBOOST_THRESHOLD:.3f}*")
        
        # View Input Summary
        with st.expander("📋 View Input Summary"):
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.markdown("**Household Characteristics**")
                st.write(f"- Country: {country}")
                st.write(f"- Residence: {residence}")
                st.write(f"- Household size: {hh_size}")
                st.write(f"- Wealth quintile: {wealth}")
                st.write(f"- Education: {education}")
                st.write(f"- Electricity: {electricity}")
                st.write(f"- Sanitation: {sanitation}")
            
            with col_sum2:
                st.markdown("**Community Context**")
                st.write(f"- Community electricity: {community_electricity}")
                st.write(f"- Community poverty: {community_poverty}")
                st.write(f"- Community education: {community_education}")
                st.write(f"- Water treatment community: {community_water_treatment}")
                st.write(f"- Media exposure community: {community_media}")
                st.write(f"- Toilet facility community: {community_toilet}")

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown("---")
st.caption("© 2026 | CatBoost Machine Learning Model for Water Access Prediction | East Africa Regional Tool")
