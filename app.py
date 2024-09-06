from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import logging
from flask_caching import Cache

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure caching
app.config['CACHE_TYPE'] = 'simple'  # Use 'redis' or other types if preferred
cache = Cache(app)

# Load the entire model pipeline
model_file_path = 'churn_prediction_model1.pkl'
numeric_values_file_path = 'numeric_unique_values.pkl'
categorical_values_file_path = 'categorical_unique_values.pkl'

try:
    with open(model_file_path, 'rb') as f:
        model_pipeline = pickle.load(f)  # Load the entire pipeline

    # Extract the preprocessor and the model from the pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    scaler = model_pipeline.named_steps['scaler']
    model = model_pipeline.named_steps['classifier']

    # Load unique values
    with open(numeric_values_file_path, 'rb') as f:
        numeric_unique_values = pickle.load(f)
    with open(categorical_values_file_path, 'rb') as f:
        categorical_unique_values = pickle.load(f)

    # Define features for prediction
    features_for_prediction = {
        'numeric': [col for col in numeric_unique_values.keys() if col != 'Churn'],
        'categorical': [col for col in categorical_unique_values.keys() if col != 'Churn']
    }
    numeric_features = features_for_prediction['numeric']
    categorical_features = features_for_prediction['categorical']

except Exception as e:
    logger.error(f"Error loading model or unique values: {e}")
    raise

@app.route('/api/v1/churnprediction', methods=['POST'])
@cache.cached(timeout=60)  # Cache the response for 60 seconds
def upload():
    # Check if the request contains a file with key 'File'
    if 'File' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['File']

    # Check if a file was selected
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is a CSV
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            # Ensure the dataframe has the necessary columns including CustomerID
            columns_needed = ['CustomerID'] + features_for_prediction['numeric'] + features_for_prediction['categorical']
            if not all(col in df.columns for col in columns_needed):
                logger.error("CSV file missing required columns")
                return jsonify({'error': 'CSV file missing required columns'}), 400

            # Preprocess the data
            df = df[columns_needed]

            # Handle missing values and convert data types
            df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
            df[numeric_features].fillna(df[numeric_features].mean(), inplace=True)
            df[categorical_features] = df[categorical_features].astype(str)
            df[categorical_features].fillna('Unknown', inplace=True)

            # Apply preprocessing and scaling
            preprocessed_data = preprocessor.transform(df.drop(columns=['CustomerID']))
            scaled_data = scaler.transform(preprocessed_data)

            # Predict
            predictions = model.predict(scaled_data)
            df['Prediction'] = predictions

            # Identify customers predicted to churn
            churn_customers = df[df['Prediction'] == 1]

            # Limit to first 6 columns (including 'Prediction')
            columns_to_display = df.columns[:6].tolist() + ['Prediction']
            df = df[columns_to_display]

            # Count churn and no churn
            churn_count = df['Prediction'].sum()
            no_churn_count = len(df) - churn_count

            # Convert DataFrame to JSON
            result_data = df.to_dict(orient='records')
            churn_customers_list = churn_customers[['CustomerID', 'Prediction']].to_dict(orient='records')

            # Create user-friendly messages
            if churn_count > no_churn_count:
                message = (
                    f"Out of {len(df)} customers, {churn_count} are predicted to churn (not continue with the business) "
                    f"and {no_churn_count} are predicted to continue with the business. "
                    "Consider implementing retention strategies for the customers at risk."
                )
            else:
                message = (
                    f"Out of {len(df)} customers, {churn_count} are predicted to churn (not continue with the business) "
                    f"and {no_churn_count} are predicted to continue with the business. "
                    "The majority of customers are predicted to stay, which is a positive sign."
                )

            churn_message = (
                f"Customers predicted to churn (CustomerID): {', '.join(map(str, churn_customers['CustomerID'].tolist()))}. "
                "These customers should be targeted for enhanced offers to improve retention."
            )

            logger.info(f"Processed {len(df)} records successfully.")
            return jsonify({
                'data': result_data,
                'churn_customers': churn_customers_list,
                'message': message,
                'churn_message': churn_message
            })

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    logger.error("Invalid file type. Only CSV files are supported.")
    return jsonify({'error': 'Invalid file type. Only CSV files are supported.'}), 400

@app.route('/')
def document():
    return render_template('document.html')

if __name__ == '__main__':
    app.run(debug=True)
