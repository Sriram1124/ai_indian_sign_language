import pickle
import numpy as np

# 1) Load the model object
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data.get('model', data)  # if it’s just the model or wrapped in a dict

# 2) Print basic info
print("Type of model:", type(model))
print("\nEstimator repr:\n", model)

# 3) Hyperparameters
print("\nHyperparameters (get_params):")
for k, v in model.get_params().items():
    print(f"  {k}: {v}")

# 4) If it’s an ensemble, show feature importances or oob_score
if hasattr(model, 'feature_importances_'):
    print("\nFeature importances (first 10):", model.feature_importances_[:10])
if hasattr(model, 'oob_score_'):
    print("Out‑of‑bag score:", model.oob_score_)

# 5) If it’s a grid‑search wrapper, show best params & best score
if hasattr(model, 'best_params_'):
    print("\nBest params from CV:", model.best_params_)
if hasattr(model, 'best_score_'):
    print("Best cross‑val score:", model.best_score_)

# 6) Classes
if hasattr(model, 'classes_'):
    print("\nClasses (labels):", model.classes_)

# # 7) How to compute accuracy yourself
# print("""
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# To get actual accuracy, you’ll need your test set:
#     from sklearn.metrics import accuracy_score
#     y_pred = model.predict(X_test)
#     print("Test accuracy:", accuracy_score(y_test, y_pred))
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# """)
