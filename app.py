import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title="ML Algorithms Explorer | Muhammad Kamil Shah",
    page_icon="📊",
    layout="wide",
)

RANDOM_STATE = 42

st.title("Machine Learning Algorithms Explorer")
st.caption("Interactive companion to the notebook-based Machine Learning Algorithms: Practical Implementation and Comparison project.")

with st.sidebar:
    st.header("Explore")
    page = st.radio(
        "Section",
        ["Project overview", "Classification", "Linear regression", "PCA"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Streamlit + scikit-learn. Uses built-in/synthetic datasets, so no API key or paid service is required.")

if page == "Project overview":
    st.subheader("What this project demonstrates")
    st.write(
        "The original repository is an educational implementation project covering a broad set of "
        "machine-learning and data-mining techniques. This app makes a focused subset interactive "
        "without pretending that every algorithm in the notebook belongs to one common benchmark."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interactive datasets", "2")
    c2.metric("Classifiers", "4")
    c3.metric("Regression models", "1")
    c4.metric("Dimensionality reduction", "PCA")

    st.markdown("### Interactive sections")
    st.markdown(
        """
        - **Classification:** compare Logistic Regression, KNN, SVM and a small neural network on Iris.
        - **Linear regression:** fit a regression line to reproducible synthetic data and inspect error metrics.
        - **PCA:** explore how much variance the first two principal components retain.
        """
    )

    st.markdown("### Broader notebook coverage")
    st.info(
        "The notebook also contains demonstrations of association-rule mining, Bayesian-network concepts, "
        "genetic algorithms and semi-supervised learning. They are documented as educational demonstrations "
        "rather than being forced into the same comparison screen."
    )

    st.markdown("### Skills demonstrated")
    st.write("Python · NumPy · pandas · scikit-learn · model evaluation · PCA · visualization · reproducible experiments")

elif page == "Classification":
    st.subheader("Iris classification")
    st.write("Choose a classifier, tune a key parameter, and compare test-set performance on the same reproducible split.")

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    model_name = st.selectbox(
        "Classifier",
        ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Neural Network (MLP)"],
    )

    if model_name == "Logistic Regression":
        c = st.slider("Regularization C", 0.1, 5.0, 1.0, 0.1)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE))
    elif model_name == "K-Nearest Neighbors":
        k = st.slider("Neighbors (k)", 1, 15, 5)
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    elif model_name == "Support Vector Machine":
        c = st.slider("Regularization C", 0.1, 5.0, 1.0, 0.1)
        model = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf"))
    else:
        hidden = st.slider("Hidden-layer neurons", 4, 32, 12)
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=1500, random_state=RANDOM_STATE),
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    left, right = st.columns([1, 1])
    with left:
        st.metric("Test accuracy", f"{acc:.1%}")
        report = classification_report(
            y_test, pred, target_names=list(iris.target_names), output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

    with right:
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(3), iris.target_names, rotation=25)
        ax.set_yticks(range(3), iris.target_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Accuracy is shown for learning/demo purposes on one fixed 70/30 split; it is not presented as a production benchmark.")

elif page == "Linear regression":
    st.subheader("Linear regression")
    st.write("Change the noise level to see how data quality affects fit and evaluation metrics.")

    noise = st.slider("Noise", 0, 60, 18)
    X, y = make_regression(
        n_samples=300, n_features=1, noise=float(noise), random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2_score(y_test, pred):.3f}")
    m2.metric("MAE", f"{mean_absolute_error(y_test, pred):.2f}")
    m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.2f}")

    order = np.argsort(X[:, 0])
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y, alpha=0.55, label="Samples")
    ax.plot(X[order, 0], model.predict(X[order]), label="Fitted line")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.set_title("Synthetic regression data")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

elif page == "PCA":
    st.subheader("Principal Component Analysis")
    st.write("Standardize Iris features, project them to two dimensions, and inspect retained variance.")

    iris = load_iris(as_frame=True)
    X = iris.data
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(scaled)

    v1, v2, vt = st.columns(3)
    v1.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]:.1%}")
    v2.metric("PC2 variance", f"{pca.explained_variance_ratio_[1]:.1%}")
    vt.metric("Total retained", f"{pca.explained_variance_ratio_.sum():.1%}")

    fig, ax = plt.subplots()
    for idx, name in enumerate(iris.target_names):
        mask = iris.target.to_numpy() == idx
        ax.scatter(projected[mask, 0], projected[mask, 1], label=name, alpha=0.75)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Iris projected to 2D")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=iris.feature_names,
        columns=["PC1", "PC2"],
    )
    st.markdown("#### Feature loadings")
    st.dataframe(loadings.round(3), use_container_width=True)

st.divider()
st.caption("Muhammad Kamil Shah · BS Data Science · Educational portfolio project")
