# 🏛️ AI/ML Engineering Practical Coding Foundations Knowledge Base

> **A Master-Level Curriculum for Building Intuition, Practical Fluency, and Deep Technical Judgment across Python, NumPy, Pandas, Scikit-Learn, PyTorch, and Production AI Pipelines.**

---

## 🗺️ Master Curriculum Roadmap

```
                          DATA FOUNDATIONS
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ▼                                                       ▼
00. How to Think with Data                              01. Python for Data & AI
     │                                                       │
     └───────────────────────────┬───────────────────────────┘
                                 ▼
                     NUMERICAL COMPUTING (NUMPY)
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
02. NumPy Foundations       03. Indexing, Slicing & Shapes 04. Broadcasting & Vectors
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 ▼
                   DATA MANIPULATION (PANDAS)
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
05. DataFrames Foundations  06. Indexing & Filtering    07. Data Cleaning & NaN
08. Feature Transformations 09. GroupBy & Aggregations  10. Merging & Reshaping
11. Time-Series Operations  12. EDA & Visualization     13. Statistical Computing
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 ▼
                   CLASSICAL MACHINE LEARNING
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
14. ML Data Workflow        15. Sklearn Estimator API   16. Preprocessing & Pipelines
17. Feature Selection & PCA 18. Training & Experiments  19. Evaluation & Metrics
20. Hyperparameters & Search     │                           │
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 ▼
                     DEEP LEARNING (PYTORCH)
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
21. PyTorch Tensors         22. Autograd & Graphs       23. Datasets & DataLoaders
24. nn.Module & Layers      25. Training Loops          26. Loss & Optimizers
27. Regularization               │                           │
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 ▼
                DEBUGGING, EXPERIMENTATION & CAPSTONE
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
28. Debugging Guide         29. Break-It Lab            30. Coding Playground
31. End-to-End Capstone Production Workflow
```

---

## 📚 Complete File Index

| File | Title | Core Concept |
| :--- | :--- | :--- |
| [**00**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/00_HOW_TO_THINK_WITH_DATA.md) | **How to Think with Data** | Rows, columns, $X$ feature matrix, $y$ target vector, dimensions, shapes, schema. |
| [**01**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/01_PYTHON_FOR_DATA_AND_AI.md) | **Python for Data and AI** | References, mutability, indexing `[::2]`, comprehensions, `*args`, `**kwargs`, lambdas. |
| [**02**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/02_NUMPY_AND_NUMERICAL_COMPUTING.md) | **NumPy Foundations** | `ndarray`, axes, dtypes, memory layouts, initialization functions. |
| [**03**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/03_ARRAY_INDEXING_SLICING_AND_SHAPES.md) | **Array Indexing, Slicing & Shapes** | `X[:, 0]`, `X[0, :]`, Boolean masks, `(N,)` vs `(N, 1)`, reshape, transpose, squeeze. |
| [**04**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/04_NUMPY_BROADCASTING_AND_VECTOR_OPERATIONS.md) | **Broadcasting & Vector Operations** | `*` vs `@`, `np.dot`, broadcasting rules `(3,4) + (4,)`, axis reduction (`axis=0` vs `axis=1`). |
| [**05**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/05_PANDAS_DATAFRAMES_FOUNDATIONS.md) | **Pandas DataFrames Foundations** | Series vs DataFrame, `df['col']` vs `df[['col']]`, schema, `.info()`, `.describe()`. |
| [**06**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/06_PANDAS_INDEXING_SELECTION_AND_FILTERING.md) | **Pandas Indexing & Filtering** | `.loc[]` vs `.iloc[]`, boolean filtering, compound conditions, SettingWithCopyWarning. |
| [**07**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/07_DATA_CLEANING_AND_MISSING_DATA.md) | **Data Cleaning & Missing Values** | `isna()`, `dropna()`, `fillna()`, imputation strategies, deduplication, type casting. |
| [**08**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/08_DATA_TRANSFORMATION_AND_FEATURE_ENGINEERING.md) | **Data Transformation & Features** | `apply()`, `map()`, `where()`, `clip()`, `cut()`, `qcut()`, One-Hot & Target encoding. |
| [**09**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/09_GROUPBY_AGGREGATION_AND_FEATURES.md) | **GroupBy, Aggregation & Features** | Split-Apply-Combine, multi-aggregations, `.agg()` vs `.transform()`, user features. |
| [**10**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/10_MERGING_JOINING_AND_RESHAPING.md) | **Merging, Joining & Reshaping** | `merge()`, `join()`, `concat()`, Inner/Left/Right/Outer, `pivot_table()`, `melt()`. |
| [**11**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/11_TIME_SERIES_DATA_OPERATIONS.md) | **Time-Series Data Operations** | `to_datetime()`, `.shift(1)` (lags), `.shift(-1)` (leads), `.rolling()`, temporal leakage. |
| [**12**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/12_DATA_VISUALIZATION_AND_EDA.md) | **Data Visualization & EDA** | Distribution analysis, boxplots, scatter matrices, correlation heatmaps, drift detection. |
| [**13**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/13_STATISTICAL_COMPUTING_IN_PYTHON.md) | **Statistical Computing in Python** | Variance with `ddof=1`, Covariance, Pearson $r$, Outliers (MAD/IQR), Student-$t$ CI. |
| [**14**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/14_MACHINE_LEARNING_DATA_WORKFLOW.md) | **ML End-to-End Data Workflow** | Complete flow: Raw data $\to X/y \to$ Temporal Split $\to$ Preprocessing $\to$ Inference. |
| [**15**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/15_SKLEARN_API_AND_ESTIMATORS.md) | **Scikit-Learn API & Estimator Design** | Estimator architecture, `fit()`, `predict()`, `predict_proba()`, `fit_transform()`. |
| [**16**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/16_PREPROCESSING_AND_PIPELINES.md) | **Preprocessing & Pipelines** | `StandardScaler`, `ColumnTransformer`, `Pipeline`, preventing preprocessor leakage. |
| [**17**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/17_FEATURE_SELECTION_AND_DIMENSIONALITY.md) | **Feature Selection & Dimensionality** | Gini importance, correlation pruning, SelectKBest, RFE, Principal Component Analysis. |
| [**18**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/18_MODEL_TRAINING_AND_EXPERIMENTS.md) | **Model Training & Experiments** | Epochs, batches, iterations, baselines, deterministic seeds, tracking. |
| [**19**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/19_MODEL_EVALUATION_AND_DIAGNOSTICS.md) | **Model Evaluation & Diagnostics** | MAE, RMSE, $R^2$, Precision, Recall, Macro $F_1$, ROC-AUC, Overfitting diagnostics. |
| [**20**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/20_HYPERPARAMETERS_AND_MODEL_SELECTION.md) | **Hyperparameters & Model Selection** | Parameters vs Hyperparameters, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit. |
| [**21**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/21_PYTORCH_TENSORS_AND_OPERATIONS.md) | **PyTorch Tensors & Operations** | NumPy to Tensor, shapes, device management (`cuda`/`cpu`), `requires_grad`. |
| [**22**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/22_PYTORCH_AUTOGRAD_AND_COMPUTATIONAL_GRAPHS.md) | **Autograd & Computational Graphs** | Dynamic graphs, forward pass, `loss.backward()`, `param.grad`, `optimizer.zero_grad()`. |
| [**23**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/23_PYTORCH_DATASETS_AND_DATALOADERS.md) | **PyTorch Datasets & DataLoaders** | Custom `Dataset` (`__len__`, `__getitem__`), `DataLoader` batching, shuffle, mini-batches. |
| [**24**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/24_PYTORCH_MODELS_AND_NN_MODULES.md) | **PyTorch Models & nn.Module** | `nn.Module`, `nn.Linear`, `BatchNorm1d`, `Dropout`, `ReLU`, `model.train()`, `model.eval()`. |
| [**25**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/25_PYTORCH_TRAINING_LOOPS.md) | **PyTorch Training Loops** | The canonical 5-step loop, validation loop, `torch.no_grad()`, early stopping. |
| [**26**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/26_LOSS_FUNCTIONS_AND_OPTIMIZERS.md) | **Loss Functions & Optimizers** | `MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`, Adam, AdamW, weight decay. |
| [**27**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/27_MODEL_REGULARIZATION_AND_GENERALIZATION.md) | **Regularization & Generalization** | Dropout, BatchNorm, $L_1$/$L_2$ weight decay, early stopping, train vs val curves. |
| [**28**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/28_DEBUGGING_DATA_AND_ML_PROBLEMS.md) | **Debugging Data & ML Problems** | Shape mismatches, NaN loss, exploding gradients, leakage debugging checklist. |
| [**29**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/29_BREAK_IT_AND_EXPERIMENTATION_LAB.md) | **Break-It & Experimentation Lab** | Adversarial stress-testing, noisy transactions, target leakage simulation. |
| [**30**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/30_AI_ML_CODING_PLAYGROUND.md) | **Interactive Coding Playground** | Step-by-step experiment exercises (Predict $\to$ Run $\to$ Observe $\to$ Explain). |
| [**31**](file:///home/pranav/Documents/AI/layer_1/AI_ML_ENGINEERING_PRACTICAL_FOUNDATIONS/31_END_TO_END_ML_ENGINEERING_WORKFLOW.md) | **End-to-End Production ML Capstone** | Complete production workflow uniting raw data, EDA, pipeline, ML, PyTorch, and CLI. |
