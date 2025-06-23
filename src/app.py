from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML model for nutrition recommendation
nutrition_model = joblib.load('models/nutrition_model.pkl')

# Symptom-to-Disease Mapping
symptoms_map = {
    'Diabetes': [
        'excessive thirst', 'frequent urination', 'fatigue',
        'blurred vision', 'slow healing wounds', 'unexplained weight loss',
        'increased hunger', 'tingling sensation in hands or feet'
    ],
    'Hypertension': [
        'headache', 'chest pain', 'vision problems',
        'shortness of breath', 'dizziness', 'nosebleeds',
        'irregular heartbeat', 'confusion'
    ],
    'Obesity': [
        'weight gain', 'snoring', 'joint pain',
        'shortness of breath', 'fatigue', 'low self-esteem',
        'difficulty sleeping', 'sweating more than usual'
    ],
    'High Cholesterol': [
        'chest pain', 'numbness', 'slurred speech',
        'yellowish deposits on eyelids (xanthelasma)', 'fatigue',
        'pain in the legs when walking', 'poor appetite'
    ]
}

required_symptoms = sorted(list(set(sym for group in symptoms_map.values() for sym in group)))

# Diet and Exercise Recommendations
disease_diet = {
    'Diabetes': 'Low sugar, low carb meals with high fiber content.',
    'Hypertension': 'Low sodium, high potassium diet rich in fruits and vegetables.',
    'Obesity': 'Calorie-deficit meals with high protein and low fat.',
    'High Cholesterol': 'Low saturated fat and high fiber diet, avoid red meat.'
}

exercise_recommendation = {
    'Diabetes': 'Brisk walking, cycling, swimming, yoga.',
    'Hypertension': 'Walking, stretching, light aerobics.',
    'Obesity': 'Cardio workouts, HIIT, walking.',
    'High Cholesterol': 'Aerobic exercises, swimming, jogging.'
}

# Meal Plan Descriptions
plan_descriptions = {
    'Keto': 'Focus on low carbs, moderate protein, high fat. Avocados, eggs, fish.',
    'Paleo': 'Whole foods like meat, fish, fruits, veggies. Avoid grains and processed foods. Promotes high protein and healthy fats from natural sources.',
    'Mediterranean': 'Balanced meals: olive oil, nuts, legumes, lean meats.',
    'Low-Fat Diet': 'Low fat, high carbs & lean protein: fruits, veggies, grains.',
    'High-Protein Diet': 'Lean protein, moderate carbs/fats for active lifestyle.'
}

def predict_disease(symptoms):
    disease_scores = {disease: 0 for disease in symptoms_map}
    for disease, disease_symptoms in symptoms_map.items():
        for symptom in symptoms:
            if symptom in disease_symptoms:
                disease_scores[disease] += 1
    return max(disease_scores, key=lambda x: (disease_scores[x], x == 'Diabetes'))

def recommend_meal(calories, protein, fat, carbs):
    if carbs < 150 and calories < 2000:
        return 'Low Carb', 'Keto'
    elif fat > 100 and protein > 150:
        return 'High Fat & Protein', 'Paleo'
    elif protein > 140 and fat < 80:
        return 'High Protein', 'High-Protein Diet'
    elif fat < 60 and 1800 < calories < 2500:
        return 'Low Fat', 'Low-Fat Diet'
    else:
        return 'Balanced', 'Mediterranean'

@app.route('/')
def home():
    return render_template('home.html', symptoms=required_symptoms)

@app.route('/result', methods=['POST'])
def result():
    name_input = request.form['name']

    if " - " in name_input:
        name, symptoms_raw = name_input.split(" - ", 1)
        selected_symptoms = [s.strip().lower() for s in symptoms_raw.split(',')]
    else:
        name = name_input
        selected_symptoms = []

    knows_disease = request.form.get('disease_status')
    disease = None

    if knows_disease == 'yes':
        disease = request.form['disease']
    else:
        if selected_symptoms:
            disease = predict_disease(selected_symptoms)
        else:
            disease = 'Unknown'

    # Collect nutrition inputs
    calories = float(request.form['calories'])
    protein = float(request.form['protein'])
    fat = float(request.form['fat'])
    carbs = float(request.form['carbs'])

    # Recommend meal plan
    smart_diet, meal_plan = recommend_meal(calories, protein, fat, carbs)

    # Predict Recommended Nutrition Targets using ML model
    try:
        model_input = pd.DataFrame([{
            'Disease': disease,
            'Meal_Plan': meal_plan
        }])
        predicted_nutrients = nutrition_model.predict(model_input)
        target_cal, target_prot, target_fat, target_carb = map(int, predicted_nutrients[0])
    except Exception as e:
        return f"Error in nutrition model prediction: {e}"

    return render_template('result.html',
        name=name,
        disease=disease,
        diet=disease_diet.get(disease, 'N/A'),
        exercise=exercise_recommendation.get(disease, 'N/A'),
        smart_diet=smart_diet,
        meal_plan=meal_plan,
        description=plan_descriptions.get(meal_plan, 'N/A'),
        input_cal=int(calories),
        input_prot=int(protein),
        input_fat=int(fat),
        input_carb=int(carbs),
        target_cal=target_cal,
        target_prot=target_prot,
        target_fat=target_fat,
        target_carb=target_carb
    )

if __name__ == '__main__':
    app.run(debug=True)
