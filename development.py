from lazyqsar.models.random_forest_binary_classifier import LazyRandomForestBinaryClassifier
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=30,
    n_redundant=0,
    weights=[0.5, 0.5],
    random_state=42
)


from sklearn.model_selection import train_test_split
# Step 2: Train the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LazyRandomForestBinaryClassifier()
clf.fit(X_train, y_train)

# Step 3: Get predicted probabilities for class 1
prob_pos = clf.predict(X_test)
print(prob_pos)

# Step 4: Calibration Curve (Reliability Diagram)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Classifier")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend()
plt.grid()
plt.show()

# Step 5: Brier Score
brier = brier_score_loss(y_test, prob_pos)
print(f"Brier score: {brier:.4f}")