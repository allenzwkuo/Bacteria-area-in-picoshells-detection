from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Placeholder storage
well_data = {}

@app.route('/upload', methods=['POST'])
def upload_image():
    well = request.form.get('well')
    time = request.form.get('time')
    concentration = request.form.get('concentration')
    image = request.files['image']

    # Here you would run the image through your ML model
    # For now, let's return a dummy percentage
    percentage = 70  # Replace with ML model output

    # Initialize list for the well if it doesn't exist
    if well not in well_data:
        well_data[well] = []

    # Check if this time point already exists
    existing_entry = next((entry for entry in well_data[well] if entry['time'] == time), None)
    
    if existing_entry:
        # Average the new percentage with the existing one
        existing_percentage = existing_entry['percentage']
        new_percentage = (existing_percentage + percentage) / 2
        existing_entry['percentage'] = new_percentage
    else:
        # If no existing entry, add a new one
        new_entry = {
            'time': time,
            'concentration': concentration,
            'percentage': percentage
        }
        well_data[well].append(new_entry)

    return jsonify({'percentage': percentage})

@app.route('/results', methods=['POST'])
def calculate_results():
    global well_data
    well_data = request.json['wellData']
    results = {}

    for well, entries in well_data.items():
        percentages = [entry['percentage'] for entry in entries]
        if all(x < y for x, y in zip(percentages, percentages[1:])):
            results[well] = 'red'

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
