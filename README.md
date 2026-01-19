🏠 House Price Prediction System
📌 Project Overview

This project is a Machine Learning–based House Price Prediction System built using Linear Regression.
The system predicts house prices based on selected features from a dataset. It demonstrates the complete ML workflow: data preprocessing, model training, evaluation, and deployment readiness.

This project was developed as part of an academic requirement.

👩‍💻 Author

Name: Naomi Egbe

Matric Number: 23CG034058

Institution: Covenant University

Course: Computer Programming / Machine Learning

🧠 Algorithm Used

Linear Regression

🗂️ Project Structure
HousePrice_Project_NaomiEgbe_23CG034058/
│
├── app.py                     # Application script
├── model_training.py          # Model development & training
├── house_price_model.pkl      # Trained regression model
├── scaler.pkl                 # StandardScaler used for preprocessing
├── dataset.csv                # Dataset used for training
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

⚙️ Technologies & Libraries

Python

Pandas

NumPy

Scikit-learn

Joblib

🔄 Workflow

Load and explore the dataset

Preprocess data (feature scaling using StandardScaler)

Split data into training and testing sets

Train a Linear Regression model

Evaluate model performance

Save trained model and scaler using Joblib

Load model in app.py for predictions

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model (Optional)
python model_training.py

3️⃣ Run the Application
python app.py

📊 Model Output

The model predicts house prices based on the input features provided.
Evaluation metrics such as Mean Absolute Error (MAE) are used to assess performance.

📝 Notes

Ensure house_price_model.pkl and scaler.pkl are present in the project directory before running app.py.

The scaler must always be applied before making predictions.

📜 License

This project is for educational purposes only.
