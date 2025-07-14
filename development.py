from lazyqsar.models.random_forest_binary_classifier import LazyRandomForestBinaryClassifier
from sklearn.datasets import make_classification


# Example usage for classification
if __name__ == "__main__":
    # Generate synthetic classification data with specified class proportions
    X, y = make_classification(
        n_samples=15000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        weights=[0.5, 0.5],
        random_state=42
    )

    # Initialize the classifier
    clf = LazyRandomForestBinaryClassifier()

    # Fit the model
    clf.fit(X, y)

    # Make predictions
    predictions = clf.predict(X)
    print("Predictions:", predictions)
