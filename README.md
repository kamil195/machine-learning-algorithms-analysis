# Machine Learning Algorithms: Practical Implementation and Comparison

An educational machine-learning project that demonstrates a broad range of algorithms and data-mining techniques in Python. The original notebook is preserved, while the optional Streamlit companion app makes a focused subset of the work easier to explore interactively.

## Interactive portfolio app

The recruiter-facing app provides four focused views:

- **Project overview** — scope, techniques, and skills demonstrated
- **Classification** — compare Logistic Regression, KNN, SVM, and a small MLP on the Iris dataset
- **Linear regression** — change synthetic-data noise and inspect R², MAE, and RMSE
- **PCA** — visualize a 2D Iris projection, explained variance, and feature loadings

The app deliberately does **not** present every notebook technique as one unified benchmark. The broader notebook contains separate educational demonstrations using different tasks and data.

## Broader notebook coverage

- Linear and logistic regression
- K-nearest neighbors and support vector machines
- Neural-network demonstrations
- Principal component analysis
- Association-rule mining
- Bayesian-network concepts
- Genetic algorithms
- Semi-supervised learning experiments

## Repository structure

```text
app.py
notebooks/
  machine_learning_algorithms.ipynb
docs/
  machine_learning_project_presentation.pptx
requirements.txt
README.md
```

## Run the interactive app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses scikit-learn built-in/synthetic datasets and requires no paid API or external service.

## Run the original notebook

```bash
jupyter notebook notebooks/machine_learning_algorithms.ipynb
```

## Portfolio note

This is an educational comparison and implementation project rather than a single production model. Some sections use synthetic or demonstration datasets. Results should be interpreted as algorithm illustrations, not as a benchmark on one unified dataset.

## Skills demonstrated

Python · NumPy · pandas · scikit-learn · model evaluation · PCA · visualization · reproducible experiments · Streamlit

## Author

**Muhammad Kamil Shah**  
BS Data Science
