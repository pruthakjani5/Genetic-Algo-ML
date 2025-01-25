import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="ML Hyperparameter Optimization",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #0066cc;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# # Create multipage navigation
# PAGES = {
#     "Home": "home",
#     "Optimizer": "optimizer",
#     "Documentation": "documentation",
#     "About": "about"
# }

class GeneticAlgorithm:
    def __init__(self, objective_function, num_variables, variable_bounds,
                population_size, max_generations, crossover_rate,
                mutation_rate, elitism_rate):
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.variable_bounds = variable_bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.elite_size = max(1, int(elitism_rate * population_size))
    
    # [Previous GA methods implementation]
    def initialize_population(self):
        return np.array([
            [np.random.uniform(low, high) for low, high in self.variable_bounds]
            for _ in range(self.population_size)
        ])
    
    def tournament_selection(self, population, fitness):
        idx = np.random.choice(len(population), size=3, replace=False)
        tournament = [(i, fitness[i]) for i in idx]
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return population[winner_idx]
    
    def roulette_selection(self, population, fitness):
        fitness = np.array(fitness)
        fitness = fitness - fitness.min() + 1e-6  # Ensure positive fitness
        probs = fitness / fitness.sum()
        idx = np.random.choice(len(population), p=probs)
        return population[idx]
    
    def rank_selection(self, population, fitness):
        ranks = np.argsort(np.argsort(fitness))
        probs = (ranks + 1) / ranks.sum()
        idx = np.random.choice(len(population), p=probs)
        return population[idx]
    
    def select_parent(self, population, fitness):
        if selection_method == "tournament":
            return self.tournament_selection(population, fitness)
        elif selection_method == "roulette":
            return self.roulette_selection(population, fitness)
        else:  # rank
            return self.rank_selection(population, fitness)
    
    def crossover(self, parent1, parent2):
        if np.random.random() > self.crossover_rate:
            return parent1
        
        if crossover_method == "single_point":
            point = np.random.randint(1, self.num_variables)
            child = np.concatenate([parent1[:point], parent2[point:]])
        elif crossover_method == "two_point":
            points = sorted(np.random.choice(range(1, self.num_variables), 2))
            child = np.concatenate([
                parent1[:points[0]],
                parent2[points[0]:points[1]],
                parent1[points[1]:]
            ])
        else:  # uniform
            mask = np.random.random(self.num_variables) < 0.5
            child = np.where(mask, parent1, parent2)
        
        return child
    
    def mutate(self, individual):
        for i in range(self.num_variables):
            if np.random.random() < self.mutation_rate:
                if mutation_method == "gaussian":
                    sigma = (self.variable_bounds[i][1] - self.variable_bounds[i][0]) * 0.1
                    individual[i] += np.random.normal(0, sigma)
                elif mutation_method == "swap":
                    j = np.random.randint(self.num_variables)
                    individual[i], individual[j] = individual[j], individual[i]
                else:  # inversion
                    individual[i] = self.variable_bounds[i][1] - individual[i]
                
                # Ensure bounds
                individual[i] = np.clip(
                    individual[i],
                    self.variable_bounds[i][0],
                    self.variable_bounds[i][1]
                )
        return individual
    
    def optimize(self):
        self.population = self.initialize_population()  # Make population accessible as instance variable
        best_solution = None
        best_fitness = float('-inf')
        history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            # Check if data is loaded in session state
            if not all(key in st.session_state for key in ['X_test', 'y_test', 'X_train', 'y_train']):
                st.error("Data not loaded. Please load the data first.")
                return None, None, []

            try:
                fitness = [self.objective_function(ind) for ind in self.population]
                
                # Update best solution
                gen_best_idx = np.argmax(fitness)
                if fitness[gen_best_idx] > best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_solution = self.population[gen_best_idx].copy()
                
                history.append(best_fitness)
            except Exception as e:
                st.error(f"Error during fitness calculation: {str(e)}")
                return None, None, []
            
            # Update progress
            progress = (generation + 1) / self.max_generations
            progress_bar.progress(progress)
            status_text.text(f"Generation {generation + 1}/{self.max_generations}")
            
            # Elitism
            elite_idx = np.argsort(fitness)[-self.elite_size:]
            elite = self.population[elite_idx]
            
            # Create new population
            new_population = []
            
            # Add elite individuals
            new_population.extend(elite)
            
            # Fill rest of population
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(self.population, fitness)
                parent2 = self.select_parent(self.population, fitness)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = np.array(new_population)
        
        progress_bar.empty()
        status_text.empty()
        return best_solution, best_fitness, history


# def load_data(name):
#     if name == "Iris":
#         data = load_iris()
#     elif name == "Wine":
#         data = load_wine()
#     elif name == "Digits":
#         data = load_digits()
#     else:  # Custom (Synthetic)
#         X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
#         return X, y, range(10)  # feature names for synthetic data
#     return data.data, data.target, data.feature_names

# def get_parameter_bounds(model_name):
#     if model_name == "Decision Tree":
#         return [(2, 20), (1, 10)]
#     elif model_name == "Random Forest":
#         return [(10, 200), (1, 10)]
#     else:  # SVM
#         return [(0.1, 10), (0.01, 1)]

# def create_model(model_name, params):
#     if model_name == "Decision Tree":
#         return DecisionTreeClassifier(
#             min_samples_split=max(int(params[0]), 2),
#             max_depth=int(params[1]) if params[1] > 0 else None
#         )
#     elif model_name == "Random Forest":
#         return RandomForestClassifier(
#             n_estimators=max(int(params[0]), 1),
#             max_depth=int(params[1]) if params[1] > 0 else None
#         )
#     else:  # SVM
#         return SVC(
#             C=max(params[0], 0.1),
#             gamma=max(params[1], 0.01)
#         )

# Main Application UI
# st.title("🧬 Genetic Algorithm for ML Hyperparameter Optimization")

# Expanded dataset options
dataset_options = {
    "Iris": "Classic flower classification",
    "Wine": "Wine quality prediction",
    "Digits": "Handwritten digit recognition",
    "Breast Cancer": "Breast cancer diagnosis",
    "Diabetes": "Diabetes progression",
    "Custom (Synthetic)": "Generated dataset with controlled properties"
}

# st.markdown("""
# This application demonstrates the power of Genetic Algorithms in optimizing machine learning model hyperparameters.
# Choose your settings below and start the optimization process.
# """)

# Expanded model options and their parameters
model_configs = {
    "Decision Tree": {
        "params": ["max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "bounds": [(1, 20), (2, 20), (1, 10), (0.1, 1.0)]
    },
    "Random Forest": {
        "params": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "bounds": [(10, 200), (1, 30), (2, 20), (1, 10), (0.1, 1.0)]
    },
    "SVM": {
        "params": ["C", "gamma", "kernel_scale", "degree"],
        "bounds": [(0.1, 100), (0.001, 10), (0.1, 10), (2, 5)]
    },
    "Gradient Boosting": {
        "params": ["n_estimators", "learning_rate", "max_depth", "min_samples_split", "subsample"],
        "bounds": [(50, 500), (0.01, 1.0), (1, 20), (2, 20), (0.5, 1.0)]
    },
    "KNN": {
        "params": ["n_neighbors", "leaf_size", "p"],
        "bounds": [(1, 50), (1, 100), (1, 5)]
    }
}
from sklearn.datasets import (
    load_iris, load_wine, load_digits, 
    load_breast_cancer, load_diabetes
)

# Expanded dataset options with descriptions
dataset_options = {
    "Iris": "Classic flower classification",
    "Wine": "Wine quality prediction",
    "Digits": "Handwritten digit recognition",
    "Breast Cancer": "Breast cancer diagnosis",
    "Diabetes": "Diabetes progression",
    "Custom (Synthetic)": "Generated dataset with controlled properties"
}

# Expanded model configurations
model_configs = {
    "Decision Tree": {
        "params": ["max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "bounds": [(1, 20), (2, 20), (1, 10), (0.1, 1.0)]
    },
    "Random Forest": {
        "params": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "bounds": [(10, 200), (1, 30), (2, 20), (1, 10), (0.1, 1.0)]
    },
    "SVM": {
        "params": ["C", "gamma", "kernel_scale", "degree"],
        "bounds": [(0.1, 100), (0.001, 10), (0.1, 10), (2, 5)]
    },
    "Gradient Boosting": {
        "params": ["n_estimators", "learning_rate", "max_depth", "min_samples_split", "subsample"],
        "bounds": [(50, 500), (0.01, 1.0), (1, 20), (2, 20), (0.5, 1.0)]
    },
    "KNN": {
        "params": ["n_neighbors", "leaf_size", "p"],
        "bounds": [(1, 50), (1, 100), (1, 5)]
    }
}

# def load_data(name):
#     """Load dataset based on name with extended options"""
#     if name == "Iris":
#         data = load_iris()
#     elif name == "Wine":
#         data = load_wine()
#     elif name == "Digits":
#         data = load_digits()
#     elif name == "Breast Cancer":
#         data = load_breast_cancer()
#     elif name == "Diabetes":
#         data = load_diabetes()
#     else:  # Custom (Synthetic)
#         X, y = make_classification(
#             n_samples=500, 
#             n_features=10, 
#             n_informative=5, 
#             random_state=42
#         )
#         return X, y, [f"Feature_{i}" for i in range(10)]
    
#     return data.data, data.target, data.feature_names
def load_data(name):
    """Load dataset based on name with extended options"""
    if name == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Assuming last column is target variable
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                feature_names = df.columns[:-1].tolist()
                return X, y, feature_names
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a CSV file")
            st.stop()
    elif name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Digits":
        data = load_digits()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    elif name == "Diabetes":
        data = load_diabetes()
    else:  # Custom (Synthetic)
        X, y = make_classification(
            n_samples=500, 
            n_features=10, 
            n_informative=5, 
            random_state=42
        )
        return X, y, [f"Feature_{i}" for i in range(10)]
    
    return data.data, data.target, data.feature_names

def create_model(model_name, params):
    """Create model with expanded options"""
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(params[0]),
            min_samples_split=int(params[1]),
            min_samples_leaf=int(params[2]),
            max_features=float(params[3])
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            min_samples_split=int(params[2]),
            min_samples_leaf=int(params[3]),
            max_features=float(params[4])
        )
    elif model_name == "SVM":
        return SVC(
            C=float(params[0]),
            gamma='scale',  # Use default 'scale' instead of manual gamma
            kernel='rbf'    # Use only RBF kernel for faster computation
        )
    elif model_name == "Gradient Boosting":
        if not hasattr(st.session_state, 'X_test'):
            st.error("Please load the data first before running optimization.")
            st.stop()
        return GradientBoostingClassifier(
            n_estimators=min(int(params[0]), 100),  # Limit max estimators
            learning_rate=float(params[1]),
            max_depth=min(int(params[2]), 5),       # Limit tree depth
            subsample=0.8   # Use fixed subsample ratio
        )
    # else:  # KNN
    elif model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=int(params[0]),
            leaf_size=int(params[1]),
            p=int(params[2])
        )


    # Sidebar content
# with st.sidebar:
#     st.title("About Genetic Algorithms in ML")
    
#     st.markdown("""
#     Genetic Algorithms (GA) are optimization techniques inspired by natural evolution. 
#     In machine learning, they help find optimal model parameters.
#     """)
    
    # Advanced GA Settings
    # with st.expander("Advanced GA Settings", expanded=False):
    #     population_diversity = st.slider("Population Diversity Factor", 0.0, 1.0, 0.5,
    #                                     help="Higher values promote more diverse solutions")
    #     adaptive_mutation = st.checkbox("Adaptive Mutation Rate", 
    #                                     help="Automatically adjust mutation rate based on population diversity")
    #     elite_competition = st.checkbox("Elite Competition",
    #                                     help="Allow elite individuals to compete with offspring")
# Set page configuration at the very start
# st.set_page_config(
#     page_title="GA-ML Optimizer",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS to improve appearance
# st.markdown("""
#     <style>
#     .main {
#         padding-top: 1rem;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #FF4B4B;
#         color: white;
#     }
#     .stButton>button:hover {
#         background-color: #FF2B2B;
#         color: white;
#     }
#     .sidebar .sidebar-content {
#         background-color: #f5f5f5;
#     }
#     div.stTitle {
#         font-weight: bold;
#         font-size: 2rem;
#         margin-bottom: 1rem;
#     }
#     .info-box {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         background-color: #f8f9fa;
#         border-left: 3px solid #FF4B4B;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Custom CSS to improve appearance with mode-independent text visibility
st.markdown("""
    <style>
    /* Common styles */
    .main {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    div.stTitle {
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Base styles that work for both modes */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #FF4B4B;
    }

    /* Light mode overrides */
    [data-theme="light"] {
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
        }
        .info-box {
            background-color: #f8f9fa;
            color: #1e1e1e;
        }
    }
    
    /* Dark mode overrides */
    [data-theme="dark"] {
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        .info-box {
            background-color: #262730;
            color: #fafafa;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Genetic Algorithm for ML Hyperparameter Optimization")

# Introduction in a nice box
st.markdown("""
    <h4>Welcome to GA-ML Optimizer!</h4>
    This application uses Genetic Algorithms to find optimal hyperparameters for machine learning models.
    Select your configuration and start optimizing!
""", unsafe_allow_html=True)

# Sidebar with better organization
with st.sidebar:
    st.title("About GA-ML")
    
    st.markdown("""
    Genetic Algorithms (GA) are optimization techniques inspired by natural evolution. 
    They excel at finding optimal solutions in complex search spaces.
    """, unsafe_allow_html=True)
    
    # Organized expandable sections
    with st.expander("🎯 Selection Methods"):
        st.markdown("""
        * **Tournament Selection**
          - Competitive selection process
          - Maintains population diversity
        
        * **Roulette Selection**
          - Fitness-proportionate selection
          - Natural evolution simulation
        
        * **Rank Selection**
          - Rank-based selection pressure
          - Prevents premature convergence
        """)
    
    with st.expander("🔄 Crossover Methods"):
        st.markdown("""
        * **Single Point**
          - Classic genetic recombination
          - Simple and effective
        
        * **Two Point**
          - Enhanced genetic diversity
          - Better trait preservation
        
        * **Uniform**
          - Maximum genetic mixing
          - Thorough space exploration
        """)
    
    with st.expander("⚡ Mutation Methods"):
        st.markdown("""
        * **Gaussian**
          - Continuous parameter tuning
          - Controlled exploration
        
        * **Swap**
          - Discrete value exchange
          - Parameter range preservation
        
        * **Inversion**
          - Gene sequence reversal
          - Local optima escape
        """)
    
    st.markdown("---")
    st.info("💡 Tip: Start with default parameters for your first optimization run!")
    st.markdown("""---""")
    st.markdown("""
    ### About the Developer

    ## *Pruthak Jani*

    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/pruthakjani5)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pruthak-jani/)

    I'm a Data Science and Machine Learning Engineering student passionate about developing innovative solutions using genetic algorithms and machine learning. This tool demonstrates the power of evolutionary algorithms in optimizing machine learning models.
    """)
    st.markdown("""---""")

# Main content in two columns with better spacing
col1, col2 = st.columns([1, 2])

# Configuration Panel with better organization
with col1:
    st.header("📋 Configuration")
    
    # Dataset and Model Selection with descriptions
    dataset_name = st.selectbox(
        "📊 Select Dataset",
        options=["Iris", "Wine", "Digits", "Custom (Synthetic)", "Breast Cancer", "Diabetes", "Upload CSV"],
        help="Choose your dataset for optimization"
    )
    
    model_name = st.selectbox(
        "🤖 Select Model",
        options=["Decision Tree", "Random Forest", "SVM", "Gradient Boosting", "KNN"],
        help="Choose your machine learning model"
    )
    model_info = {
        "Decision Tree": """
            Parameters to optimize:
            - min_samples_split: Minimum samples required to split
            - max_depth: Maximum depth of the tree
            """,
        "Random Forest": """
            Parameters to optimize:
            - n_estimators: Number of trees
            - max_depth: Maximum depth of trees
            """,
        "SVM": """
            Parameters to optimize:
            - C: Regularization parameter
            - gamma: Kernel coefficient
            """,
        "Gradient Boosting": """
            Parameters to optimize:
            - n_estimators: Number of boosting stages
            - learning_rate: Boosting rate
            - max_depth: Maximum depth of trees
            """,
        "KNN": """
            Parameters to optimize:
            - n_neighbors: Number of neighbors
            - leaf_size: Leaf size for tree
            - p: Power parameter for Minkowski distance
            """
    }
    st.info(model_info[model_name])
    
    # Display model information in a nice box
    # st.markdown(f"""
    #     <div class="info-box">
    #     <strong>{model_name} Parameters:</strong><br>
    #     {model_info[model_name]}
    #     </div>
    # """, unsafe_allow_html=True)
    
    # GA Parameters in organized sections
    st.subheader("⚙️ GA Parameters")
    with st.container():
        population_size = st.slider("Population Size", 10, 100, 20)
        max_generations = st.slider("Generations", 10, 200, 50)
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)
        elitism_rate = st.slider("Elitism Rate", 0.0, 0.3, 0.1)
    
    # GA Strategies with better organization
    st.subheader("🔧 GA Strategies")
    with st.container():
        crossover_method = st.selectbox("Crossover Type", ["single_point", "two_point", "uniform"])
        selection_method = st.selectbox("Selection Type", ["tournament", "roulette", "rank"])
        mutation_method = st.selectbox("Mutation Type", ["gaussian", "swap", "inversion"])

# [Rest of your existing code for objective_function and main content panel remains the same]
# Define objective function
def objective_function(params):
    model = create_model(model_name, params)
    model.fit(st.session_state.X_train, st.session_state.y_train)
    y_pred = model.predict(st.session_state.X_test)
    return accuracy_score(st.session_state.y_test, y_pred)

# Main Content Panel
with col2:
    st.header("Optimization Process")
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load and prepare data
    if st.button("Load Data", type="primary"):
        X, y, feature_names = load_data(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_names
        st.session_state.data_loaded = True
        
        st.success("Data loaded successfully!")
        st.write("Dataset shape:", X.shape)
    
    if st.session_state.data_loaded:
        if st.button("Start Optimization", type="primary"):
            st.write("### Optimization Progress")
            
            variable_bounds = model_configs[model_name]["bounds"]
            
            # Initialize and run GA
            ga = GeneticAlgorithm(
                objective_function=objective_function,
                num_variables=len(variable_bounds),
                variable_bounds=variable_bounds,
                population_size=population_size,
                max_generations=max_generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elitism_rate=elitism_rate
            )
            
            # Run optimization and get final population
            best_params, best_fitness, history = ga.optimize()
            final_population = ga.population  # Store final population
            
            # Create final model with best parameters
            best_model = create_model(model_name, best_params)
            best_model.fit(st.session_state.X_train, st.session_state.y_train)
            final_predictions = best_model.predict(st.session_state.X_test)
            
            # Display Results
            st.success("Optimization Complete!")
            
            # Results tabs
            tab1, tab2 = st.tabs(["Performance Metrics", "Visualizations"])
            
            with tab1:
                st.metric("Best Accuracy", f"{best_fitness:.4f}")
                
                param_names = {
                    "Decision Tree": ["min_samples_split", "max_depth"],
                    "Random Forest": ["n_estimators", "max_depth"],
                    "SVM": ["C", "gamma"],
                    "Gradient Boosting": ["n_estimators", "learning_rate", "max_depth"],
                    "KNN": ["n_neighbors", "leaf_size", "p"]
                }[model_name]
                
                params = model_configs[model_name]['params'][:len(best_params)]
                values = [round(float(x), 4) for x in best_params[:len(params)]]
                param_df = pd.DataFrame({
                    'Parameter': params,
                    'Value': values
                })
                st.table(param_df)
                
                st.text("Classification Report:")
                st.text(classification_report(st.session_state.y_test, final_predictions))
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Optimization History
                    fig = px.line(
                        x=range(len(history)),
                        y=history,
                        labels={'x': 'Generation', 'y': 'Best Fitness'},
                        title='Fitness Evolution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confusion Matrix
                    cm = confusion_matrix(st.session_state.y_test, final_predictions)
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Additional Visualizations in a separate container
                st.subheader("📊 Additional Visualizations and Metrics")
                if 'data_loaded' in st.session_state and st.session_state.data_loaded:
                    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                        "Learning Curves", 
                        "Feature Importance", 
                        "Model Comparison",
                        "Performance Metrics"
                    ])
                    
                    with viz_tab1:
                        # Learning Curves
                        train_sizes, train_scores, test_scores = learning_curve(
                            best_model, st.session_state.X_train, st.session_state.y_train,
                            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
                        )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=train_sizes, 
                            y=train_scores.mean(axis=1),
                            mode='lines+markers', 
                            name='Training Score',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=train_sizes, 
                            y=test_scores.mean(axis=1),
                            mode='lines+markers', 
                            name='Cross-Validation Score',
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            title='Learning Curves',
                            xaxis_title='Training Examples',
                            yaxis_title='Score',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab2:
                        # Feature Importance
                        if hasattr(best_model, 'feature_importances_'):
                            importances = best_model.feature_importances_
                            feature_imp = pd.DataFrame({
                                'Feature': st.session_state.feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feature_imp,
                                x='Feature',
                                y='Importance',
                                title='Feature Importance',
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Feature importance not available for this model type.")
                    
                    with viz_tab3:
                        # Model Comparison
                        models = {
                            'Decision Tree': DecisionTreeClassifier(),
                            'Random Forest': RandomForestClassifier(),
                            'SVM': SVC(),
                            'KNN': KNeighborsClassifier()
                        }
                        
                        comparison_scores = []
                        for model_type, model in models.items():
                            scores = cross_val_score(model, st.session_state.X_train, st.session_state.y_train, cv=5)
                            comparison_scores.append({
                                'Model': model_type,
                                'Mean Score': scores.mean(),
                                'Std Score': scores.std()
                            })
                        
                        comparison_df = pd.DataFrame(comparison_scores)
                        fig = px.bar(
                            comparison_df,
                            x='Model',
                            y='Mean Score',
                            error_y='Std Score',
                            title='Model Comparison',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab4:
                        # Performance Metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            precision = precision_score(st.session_state.y_test, final_predictions, average='weighted')
                            recall = recall_score(st.session_state.y_test, final_predictions, average='weighted')
                            f1 = f1_score(st.session_state.y_test, final_predictions, average='weighted')
                            
                            st.metric("Precision", f"{precision:.4f}")
                            st.metric("Recall", f"{recall:.4f}")
                            st.metric("F1 Score", f"{f1:.4f}")
                        
                        with metric_col2:
                            try:
                                if hasattr(best_model, 'predict_proba'):
                                    y_proba = best_model.predict_proba(st.session_state.X_test)
                                    log_loss_score = log_loss(st.session_state.y_test, y_proba)
                                    st.metric("Log Loss", f"{log_loss_score:.4f}")
                                
                                # For ROC AUC, try to compute if possible
                                try:
                                    if hasattr(best_model, 'predict_proba'):
                                        y_proba = best_model.predict_proba(st.session_state.X_test)
                                        if len(np.unique(st.session_state.y_test)) == 2:
                                            roc_auc = roc_auc_score(st.session_state.y_test, y_proba[:, 1])
                                        else:
                                            roc_auc = roc_auc_score(st.session_state.y_test, y_proba, multi_class='ovr')
                                        st.metric("ROC AUC", f"{roc_auc:.4f}")
                                    else:
                                        st.info("ROC AUC not available for this model type.")
                                except Exception as e:
                                    st.info("ROC AUC calculation not possible with current configuration.")
                            except Exception as e:
                                st.info("Some metrics not available for this model type.")
                        
                        with metric_col3:
                            kappa = cohen_kappa_score(st.session_state.y_test, final_predictions)
                            mcc = matthews_corrcoef(st.session_state.y_test, final_predictions)
                            st.metric("Cohen's Kappa", f"{kappa:.4f}")
                            st.metric("Matthews Correlation", f"{mcc:.4f}")


# Footer with better styling
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Made with ❤️ by Pruthak Jani</p>
    <p style='font-size: 0.8em;'>Version 1.0.0 | @ 2024</p>
    </div>
""", unsafe_allow_html=True)
