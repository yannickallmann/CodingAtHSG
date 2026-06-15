

# Luxury Watch Price Estimator

A data-driven tool to estimate the fair market value of any luxury watch, built as the group project for Skills: Introduction to Programming at the University of St. Gallen (HSG).

**Authors:** Jan Nemeth, Daniel Kullmann, Timon Mata, Julian Kreiliger, Philipp Sajthy, Yannick Allmann

------------------------------------------------------------------------

## What We Are Doing

The luxury watch market is opaque and fragmented. Identical watches list at different prices across dealers, and a private seller has almost no objective way to know what their watch is worth.

Therefore this project builds a machine learning model that predicts the resale price of a luxury watch based on its core specifications: brand, model, year of production, case material, bracelet material, movement, size, condition, and intended wearer.

We work with a Kaggle dataset of luxury watch listings, clean it, explore it, and train an XGBoost regression model on the log-transformed price. The end product is an interactive Streamlit web app that lets a user enter the specs of any watch and instantly receive an estimated market value.

------------------------------------------------------------------------

## Project Structure — Why Everything Exists Twice

You will notice that the code appears in two places. This is intentional:

**The notebook (`LuxuryWatches.ipynb`)** is the story. It walks through every step of the project: data loading, cleaning, exploration, modelling, and evaluation with full explanations and visualizations at each stage. It is meant to be read and graded. It shows how we progressed through the project.

**The source files (`src/`)** are the engine. The same logic is re-implemented as clean, documented Python classes that power the Streamlit app and the training script. This separation means the app can run without anyone ever opening Jupyter. All you need to do to run it is to run `train.py` to train the model and then start the app via `app.py`.

In short: the notebook is hand-written step by step so the whole analysis stays transparent and reproducible for the reader, while the `src/` classes mirror that same logic in a form built for deployment. The two are maintained in parallel and kept in sync by hand.

------------------------------------------------------------------------

## How to Run the Project

### 1. Clone the repository and install dependencies

This project requires **Python ≥ 3.11**. The commands work on Windows, macOS, and Linux — use the block for your system.

**Windows (PowerShell):**

``` powershell
git clone https://github.com/yannickallmann/CodingAtHSG.git
cd CodingAtHSG
pip install -r requirements.txt
```

**macOS / Linux (Terminal):**

``` bash
git clone https://github.com/yannickallmann/CodingAtHSG.git
cd CodingAtHSG
pip3 install -r requirements.txt
```

> **macOS only:** XGBoost needs the OpenMP runtime, which is not bundled with the pip wheel. If `import xgboost` later fails with a `libomp.dylib` error, install it once with `brew install libomp`.

### 2. Train and save the model

This step trains the XGBoost model and saves it to `models/watch_price_model.joblib`. It takes roughly 20 minutes depending on your machine, and you only need to do it once.

**Windows:**

``` powershell
python train.py
```

**macOS / Linux:**

``` bash
python3 train.py
```

### 3. Launch the Streamlit app

**Windows:**

``` powershell
python -m streamlit run app.py
```

**macOS / Linux:**

``` bash
python3 -m streamlit run app.py
```

We use `python -m streamlit ...` because it works as long as you use the same Python you installed the requirements with; if `streamlit` is already on your PATH, plain `streamlit run app.py` works too. Run the command from the repository root so the app finds `src/` and `models/`. The estimator opens in your browser at `http://localhost:8501` (open the URL manually if it doesn't). Select a brand, model, and watch specifications, then click "Estimate Price". Stop the app with `Ctrl+C` in the terminal.

### 4 Run the notebook

Open `LuxuryWatches.ipynb` in Jupyter or VS Code and run all cells top to bottom. The hyperparameter tuning step (Part 6) takes roughly 5–10 minutes.

------------------------------------------------------------------------

## About the Notebook

### Part 1: Dataset Loading

We load the Kaggle dataset of \~284,000 luxury watch listings and inspect the raw data.

### Part 2: Data Cleaning — Missing Values

We quantify missing data, drop uninformative columns, and consolidate the two redundant condition columns without losing rows. We also document the selection bias introduced by dropping "Price on request" entries and rows with missing prices — the cleaned dataset skews towards liquid, well-documented, mid-range watches.

### Part 3: Data Cleaning — Data Types

We parse raw strings into proper numeric types: case size (mm), year of production, and price. All cleaning is vectorized for performance.

### Part 4: Exploratory Data Analysis

We explore distributions and relationships visually. Key findings: price is heavily right-skewed (log transformation needed), brand and material are strong price drivers, size and year of production matter less in isolation.

### Part 5: Data Preprocessing

We apply the log transformation to the price target and prepare categorical variables. Missing categoricals are filled with "Unknown" so the model treats the absence of information as its own signal. The target encoder for high-cardinality variables (brand, model) is placed inside the cross-validation pipeline to prevent data leakage.

### Part 6: Machine Learning

Three-step process: baselSines (Dummy + Ridge), XGBoost pipeline with target encoding and hyperparameter tuning via RandomizedSearchCV, and final evaluation using MAPE. The tuned model achieves a MAPE of 30.3%, compared to 216.7% for the naive baseline and 40.3% for Ridge.

Visualizations include a MAPE comparison chart, a SHAP waterfall plot for a single prediction, and a feature importance chart. These are interpreted carefully — brand and model dominate importance scores partly because they carry price information through target encoding, not necessarily because of independent causal effects.

### Part 7: Interactive Price Estimator

A notebook-based prompt that walks the user through entering watch specs and returns an estimated market value. Useful for quick testing without launching the full Streamlit app.

------------------------------------------------------------------------

## Files in This Repository

| File/Folder | Purpose |
|----------------------|--------------------------------------------------|
| `LuxuryWatches.ipynb` | Main notebook — the full story with explanations. |
| `src/cleaner.py` | `WatchDataCleaner` class — loads and cleans raw data. |
| `src/eda.py` | `WatchEDA` class — all exploratory visualizations. |
| `src/model.py` | `WatchPriceModel` class — train, evaluate, save, predict. |
| `train.py` | One-time script to train and save the model. |
| `app.py` | Streamlit web app — the interactive price estimator. |
| `data/Watches.csv` | Raw dataset (\~40 MB). |
| `data/Watches.csv.zip` | Same dataset, zipped (\~7 MB). |
| `models/` | Saved trained model (generated by `train.py`). |
| `requirements.txt` | Python packages needed to run the project. |
| `AI_reflection.md` | Reflection on how AI tools were used during the project. |
| `README.md` | This file. |

------------------------------------------------------------------------

## Notes & Limitations

- **Selection bias**: the cleaned dataset excludes "Price on request" listings and watches with missing prices. These tend to be rare, illiquid, or ultra-high-end pieces. The model is most reliable for mainstream luxury watches in the \$2,000–\$50,000 range.

- **Condition paradox**: changing condition alone (e.g. Good → Poor) may not always move the price in the expected direction. This is because brand and model dominate the model's predictions, and condition's effect is relative — a Poor Rolex Daytona still fetches more than a Mint Seiko.

- **Target encoding**: brand and model are encoded using their average historical price. Their high feature importance is partly a reflection of this encoding, not evidence of independent causal effects.

- **Asking prices vs. transaction prices**: the dataset reflects listed prices, not confirmed sales. The estimator predicts a fair listing price rather than a guaranteed sale price.

## How Well Does It Work in Practice?

Tested against real secondary market listings, the model performs well for standard, liquid watches. Performance degrades predictably for edge cases — unusual material combinations, very rare conditions like "Unworn" or "Poor", and made-up brand/model combinations that were never in the training data. This is expected and honest: it is fundamentally impossible to price something that has never been sold before. The model can only interpolate within what it has seen - it cannot extrapolate into uncharted territory.
