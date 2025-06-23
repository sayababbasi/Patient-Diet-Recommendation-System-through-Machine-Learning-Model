
import sys
import joblib
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout,
    QPushButton, QCheckBox, QTextEdit, QFormLayout, QHBoxLayout, QMessageBox
)

# Load models and preprocessing tools
disease_model = joblib.load('models/disease_rec_model.pkl')
meal_model = joblib.load('models/meal_recommend_model.pkl')
nutrition_model = joblib.load('models/nutrition_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_selector = joblib.load('models/feature_selector.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
encoders = joblib.load('models/encoders.pkl')  # contains 'symptoms' list

class DietRecommendationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diet & Disease Recommendation")
        self.setFixedWidth(500)
        self.symptom_list = encoders['symptoms']
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.calories_input = QLineEdit()
        self.protein_input = QLineEdit()
        self.fat_input = QLineEdit()
        self.carbs_input = QLineEdit()

        form_layout.addRow("Name:", self.name_input)
        form_layout.addRow("Calories:", self.calories_input)
        form_layout.addRow("Protein:", self.protein_input)
        form_layout.addRow("Fat:", self.fat_input)
        form_layout.addRow("Carbs:", self.carbs_input)

        layout.addLayout(form_layout)

        layout.addWidget(QLabel("Select Symptoms:"))
        self.symptom_checkboxes = []
        for symptom in self.symptom_list:
            cb = QCheckBox(symptom.replace("_", " ").title())
            self.symptom_checkboxes.append(cb)
            layout.addWidget(cb)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        btn_layout = QHBoxLayout()
        self.predict_button = QPushButton("Get Recommendations")
        self.clear_button = QPushButton("Clear")
        btn_layout.addWidget(self.predict_button)
        btn_layout.addWidget(self.clear_button)
        layout.addLayout(btn_layout)

        self.predict_button.clicked.connect(self.make_predictions)
        self.clear_button.clicked.connect(self.clear_form)

        self.setLayout(layout)

    def make_predictions(self):
        try:
            name = self.name_input.text()
            symptoms = [1 if cb.isChecked() else 0 for cb in self.symptom_checkboxes]
            nutrition = [
                float(self.calories_input.text()),
                float(self.protein_input.text()),
                float(self.fat_input.text()),
                float(self.carbs_input.text())
            ]

            # Predict disease
            X_symptoms = pd.DataFrame([symptoms], columns=self.symptom_list)
            X_symptoms = feature_selector.transform(X_symptoms)
            disease = disease_model.predict(X_symptoms)[0]
            disease = label_encoder.inverse_transform([disease])[0]

            # Predict meal plan
            scaled_nutrition = scaler.transform([nutrition])
            meal = meal_model.predict(scaled_nutrition)[0]

            # Predict nutritional targets
            nutrition_pred = nutrition_model.predict(scaled_nutrition)[0]
            nutrition_result = {
                'Calories': round(nutrition_pred[0], 2),
                'Protein': round(nutrition_pred[1], 2),
                'Fat': round(nutrition_pred[2], 2),
                'Carbs': round(nutrition_pred[3], 2)
            }

            # Show results
            self.result_box.setText(
                f"👤 Name: {name}\n\n"
                f"🦠 Predicted Disease: {disease}\n"
                f"🍱 Recommended Meal Plan: {meal}\n\n"
                f"🎯 Nutritional Targets:\n"
                + "\n".join([f"  - {k}: {v}" for k, v in nutrition_result.items()])
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Something went wrong:\n{str(e)}")

    def clear_form(self):
        self.name_input.clear()
        self.calories_input.clear()
        self.protein_input.clear()
        self.fat_input.clear()
        self.carbs_input.clear()
        for cb in self.symptom_checkboxes:
            cb.setChecked(False)
        self.result_box.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DietRecommendationApp()
    window.show()
    sys.exit(app.exec_())
