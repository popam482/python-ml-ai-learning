# Python ML & AI Learning

Personal learning repository for Python, Artificial Intelligence, and Machine Learning concepts.

The repository collects short, focused examples and exercises that document core data-science tooling and workflows — from NumPy fundamentals through data manipulation, model training, evaluation, and experiments with real datasets.

## Table of Contents

* [Project Overview](#project-overview)
* [Stack and Notable Libraries](#stack-and-notable-libraries)
* [Learning Roadmap and Approach](#learning-roadmap-and-approach)
* [Example Projects and Demonstrations](#example-projects-and-demonstrations)

  * [ML Playground / Projects with Real Datasets](#ml-playground--projects-with-real-datasets)
  * [Representative Scripts](#representative-scripts)
* [Project Layout](#project-layout)
* [How to Run It](#how-to-run-it)
* [Recommendations for Reproducible Experiments](#recommendations-for-reproducible-experiments)
* [Contributing](#contributing)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)

## Project Overview

This repository is organized as a hands-on learning playground.

Each numbered folder (`01-numpy` ... `13-deep_learning`) focuses on a specific topic or technique, providing small, runnable Python scripts that demonstrate concepts and typical workflows used in data analysis and machine learning.

The repository is intended primarily for learning and experimentation rather than production use.

## Stack and Notable Libraries

### Language

* **Python 100%**

### Runtime / Environment

* CPython 3.8+
* Jupyter / JupyterLab for interactive exploration

### Notable Libraries

* **NumPy** — numerical arrays and vectorized computation
* **pandas** — tabular data manipulation and cleaning
* **Matplotlib / Seaborn** — plotting and data visualization
* **scikit-learn** — classical machine learning algorithms, preprocessing, and evaluation
* **TensorFlow / Keras or PyTorch** — introductory deep-learning experimentation

## Learning Roadmap and Approach

The repository is structured for progressive learning:

1. **Foundations** — vectorized operations, broadcasting, indexing, and array manipulation with NumPy (`01-numpy`).
2. **Tabular Data** — loading, cleaning, grouping, merging, and pivoting with pandas (`02-pandas`).
3. **Visualization** — building plots and exploratory charts with Matplotlib and Seaborn (`03-matplot`).
4. **Feature Engineering** — vectorization, broadcasting, and other numerical techniques (`04-vectorization`, `05-broadcasting`, `06-numpy_vectors`).
5. **Supervised Learning** — train/test splitting, linear regression, logistic regression, and model evaluation (`07-sklearn`, `08-Linear_Regression`, `09-Logistic_Regression`, `10-train_test_split`, `11-confusion_matrix`).
6. **Unsupervised Learning** — clustering, dimensionality reduction, and related techniques (`12-unsupervised_learning`).
7. **Deep Learning** — introductory neural-network experiments (`13-deep_learning`).
8. **End-to-End Projects** — applying the learned concepts to real datasets and complete workflows (`projects-with-real-datasets`, `datasets`).

The learning process favors small, focused scripts with clear outputs and comments.

Each folder is intended to function as a short, self-contained lesson: read the script, run it, and experiment by changing inputs, parameters, or implementation details.

## Example Projects and Demonstrations

### ML Playground / Projects with Real Datasets

The repository also contains projects that combine multiple concepts into more complete workflows.

These projects typically cover:

* Data loading and preprocessing
* Exploratory data analysis
* Feature engineering
* Model training
* Model evaluation
* Data visualization
* Experimentation with different features and model parameters

The goal is to gradually move from isolated examples toward end-to-end machine-learning workflows using real-world datasets.

### Representative Scripts

Some examples include:

#### NumPy

`01-numpy/01_numpy_learning.py`

* Array creation and manipulation
* Reshaping
* Indexing and slicing
* Broadcasting
* Aggregation
* Vectorized operations

`01-numpy/01_numpy_sensor_pulse_analytics.py`

* Synthetic sensor-data generation
* Data cleaning
* Normalization
* Numerical analysis
* Simple composite health-index calculations

#### pandas

`02-pandas/02_pandas_learning.py`

* DataFrame creation
* Selection and filtering
* Grouping and aggregation
* Joins
* Pivot tables
* Missing-value handling

Other numbered folders contain similar focused examples covering machine-learning algorithms, preprocessing, evaluation, visualization, and deep-learning concepts.

## Project Layout

```text
python-ml-ai-learning/
│
├── 01-numpy/
│   ├── 01_numpy_learning.py
│   └── 01_numpy_sensor_pulse_analytics.py
│
├── 02-pandas/
│   └── 02_pandas_learning.py
│
├── 03-matplot/
│   └── ...
│
├── 04-vectorization/
│   └── ...
│
├── 05-broadcasting/
│   └── ...
│
├── 06-numpy_vectors/
│   └── ...
│
├── 07-sklearn/
│   └── ...
│
├── 08-Linear_Regression/
│   └── ...
│
├── 09-Logistic_Regression/
│   └── ...
│
├── 10-train_test_split/
│   └── ...
│
├── 11-confusion_matrix/
│   └── ...
│
├── 12-unsupervised_learning/
│   └── ...
│
├── 13-deep_learning/
│   └── ...
│
├── datasets/
│   └── ...
│
├── projects-with-real-datasets/
│   └── ...
│
├── exports/
│   └── ...
│
├── .gitignore
└── python-ml-ai-playground.iml
```

### Folder Overview

| Folder                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `01-numpy/`                    | NumPy exercises and numerical operations                     |
| `02-pandas/`                   | pandas examples for data manipulation and analysis           |
| `03-matplot/`                  | Matplotlib and Seaborn visualization examples                |
| `04-vectorization/`            | Vectorization techniques and performance-related examples    |
| `05-broadcasting/`             | NumPy broadcasting and advanced array operations             |
| `06-numpy_vectors/`            | Vectorized workflows for numerical tasks                     |
| `07-sklearn/`                  | scikit-learn examples and model-fitting workflows            |
| `08-Linear_Regression/`        | Linear regression demonstrations and diagnostics             |
| `09-Logistic_Regression/`      | Logistic regression demonstrations and evaluation            |
| `10-train_test_split/`         | Train/test splitting and related validation concepts         |
| `11-confusion_matrix/`         | Classification evaluation and confusion matrices             |
| `12-unsupervised_learning/`    | Clustering, PCA, and related techniques                      |
| `13-deep_learning/`            | Introductory deep-learning experiments                       |
| `datasets/`                    | Raw or small sample datasets used by examples                |
| `projects-with-real-datasets/` | End-to-end projects using publicly available datasets        |
| `exports/`                     | Generated plots, model outputs, and other exported artifacts |


## How to Run It

### 1. Clone the Repository

```bash
git clone https://github.com/popam482/python-ml-ai-learning.git
cd python-ml-ai-learning
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Core Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

If a `requirements.txt` file is added later:

```bash
python -m pip install -r requirements.txt
```

### 4. Run an Example

For example:

```bash
python 01-numpy/01_numpy_learning.py
```

```bash
python 01-numpy/01_numpy_sensor_pulse_analytics.py
```

```bash
python 02-pandas/02_pandas_learning.py
```

### 5. Open JupyterLab

For interactive examples:

```bash
jupyter lab
```

