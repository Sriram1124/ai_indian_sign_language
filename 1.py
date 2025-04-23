# import joblib
# import pprint
# import numpy as np

# # Load the model
# model = joblib.load('random_forest_model.pkl')

# # Check type
# print("✅ Model type:", type(model))

# # Show estimator (basic)
# print("\n🔍 Estimator Representation:\n", model)

# # Show hyperparameters
# print("\n⚙️ Hyperparameters:")
# pprint.pprint(model.get_params())

# # Feature importance (first 10, if available)
# if hasattr(model, "feature_importances_"):
#     print("\n📊 Feature Importances (top 10):")
#     print(model.feature_importances_[:10])

# # Class labels
# if hasattr(model, "classes_"):
#     print("\n🏷️ Classes:", model.classes_)

# # If training score exists
# if hasattr(model, "score") and hasattr(model, "X_train_"):  # custom models might have X_train_
#     try:
#         print("\n📈 Training Accuracy:", model.score(model.X_train_, model.y_train_))
#     except:
#         print("\n⚠️ Training data not stored in model. Cannot compute training score.")

import joblib
label_encoder = joblib.load('label_encoder.pkl')
print("Class labels:", label_encoder.classes_)  # This maps 0–12 to real labels like 'Hello', 'Cinema', etc.
