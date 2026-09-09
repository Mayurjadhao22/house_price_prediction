import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template_string, request

# -----------------------------------------------------------------------------
# NumPy 2.x to 1.x Unpickling Compatibility Layer for Vercel Serverless
# -----------------------------------------------------------------------------
try:
    import numpy._core.multiarray
except ImportError:
    import numpy.core.multiarray
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# -----------------------------------------------------------------------------
# UI Layout Template
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Valuation Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: #0f172a;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.2) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.2) 0px, transparent 50%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            color: #f8fafc;
        }

        .container {
            width: 100%;
            max-width: 720px;
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.6),
                0 8px 10px -6px rgba(0, 0, 0, 0.4),
                0 0 50px rgba(99, 102, 241, 0.15);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.025em;
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        select option {
            background-color: #0f172a;
            color: #ffffff;
        }

        input:focus, select:focus {
            border-color: #818cf8;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.2),
                0 0 0 3px rgba(129, 140, 248, 0.25),
                0 4px 12px rgba(129, 140, 248, 0.15);
        }

        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 14px rgba(99, 102, 241, 0.4),
                0 0 20px rgba(168, 85, 247, 0.25);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 6px 20px rgba(99, 102, 241, 0.6),
                0 0 30px rgba(168, 85, 247, 0.45);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
            border: 1px solid rgba(168, 85, 247, 0.35);
            border-radius: 16px;
            text-align: center;
            box-shadow: 
                0 10px 25px -5px rgba(168, 85, 247, 0.25),
                inset 0 1px 1px rgba(255, 255, 255, 0.1);
            animation: fadeIn 0.4s ease-out;
        }

        .result-box h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #c084fc;
            margin-bottom: 0.5rem;
        }

        .result-box .value {
            font-size: 2.25rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 0 12px rgba(255, 255, 255, 0.3);
        }

        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            color: #fca5a5;
            text-align: center;
            font-size: 0.9rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 640px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, .btn-submit {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Property Valuation</h1>
            <p>Enter physical attributes to calculate estimated valuation</p>
        </div>

        <form method="POST" action="/" class="grid-form">
            <div class="form-group">
                <label for="Square_Footage">Square Footage</label>
                <input type="number" step="any" id="Square_Footage" name="Square_Footage" value="{{ inputs.get('Square_Footage', '1800') }}" required>
            </div>

            <div class="form-group">
                <label for="Num_Bedrooms">Number of Bedrooms</label>
                <input type="number" step="1" id="Num_Bedrooms" name="Num_Bedrooms" value="{{ inputs.get('Num_Bedrooms', '3') }}" required>
            </div>

            <div class="form-group">
                <label for="Num_Bathrooms">Number of Bathrooms</label>
                <input type="number" step="1" id="Num_Bathrooms" name="Num_Bathrooms" value="{{ inputs.get('Num_Bathrooms', '2') }}" required>
            </div>

            <div class="form-group">
                <label for="Year_Built">Year Built</label>
                <input type="number" step="1" id="Year_Built" name="Year_Built" value="{{ inputs.get('Year_Built', '2015') }}" required>
            </div>

            <div class="form-group">
                <label for="Lot_Size">Lot Size (sq ft)</label>
                <input type="number" step="any" id="Lot_Size" name="Lot_Size" value="{{ inputs.get('Lot_Size', '5000') }}" required>
            </div>

            <div class="form-group">
                <label for="Garage_Size">Garage Capacity (Cars)</label>
                <input type="number" step="1" id="Garage_Size" name="Garage_Size" value="{{ inputs.get('Garage_Size', '2') }}" required>
            </div>

            <div class="form-group full-width">
                <label for="Neighborhood_Quality">Neighborhood Quality Category</label>
                <select id="Neighborhood_Quality" name="Neighborhood_Quality" required>
                    <option value="1" {% if inputs.get('Neighborhood_Quality') == '1' %}selected{% endif %}>Low (Tier 1)</option>
                    <option value="2" {% if inputs.get('Neighborhood_Quality') == '2' or not inputs %}selected{% endif %}>Medium (Tier 2)</option>
                    <option value="3" {% if inputs.get('Neighborhood_Quality') == '3' %}selected{% endif %}>High (Tier 3)</option>
                    <option value="4" {% if inputs.get('Neighborhood_Quality') == '4' %}selected{% endif %}>Premium (Tier 4)</option>
                </select>
            </div>

            <button type="submit" class="btn-submit">Calculate Valuation</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h3>Predicted Estimated Value</h3>
            <div class="value">${{ "{:,.2f}".format(prediction) }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    inputs = {}

    if request.method == 'POST':
        inputs = request.form.to_dict()
        if model is None:
            error = "Model file 'linear.pkl' failed to load properly."
        else:
            try:
                # Features map to original dataset order[cite: 3]:
                # 1. Square_Footage, 2. Num_Bedrooms, 3. Num_Bathrooms
                # 4. Year_Built, 5. Lot_Size, 6. Garage_Size, 7. Neighborhood_Quality
                features = [
                    float(request.form.get('Square_Footage', 0)),
                    float(request.form.get('Num_Bedrooms', 0)),
                    float(request.form.get('Num_Bathrooms', 0)),
                    float(request.form.get('Year_Built', 0)),
                    float(request.form.get('Lot_Size', 0)),
                    float(request.form.get('Garage_Size', 0)),
                    float(request.form.get('Neighborhood_Quality', 1))
                ]
                
                pred_val = model.predict(np.array([features]))[0]
                prediction = float(pred_val)
            except Exception as e:
                error = f"Prediction Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error, inputs=inputs)

# Required entry point for Vercel WSGI Handler
app = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
  





   
