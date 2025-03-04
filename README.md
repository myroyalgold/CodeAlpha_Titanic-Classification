# CODEALPHA_TITATANIC CLASSIFICATION PROJECT

## PROBLEM STATEMENT: 
Make a system which tells whether the person will be save from sinking. What factors were most likely lead to success-socio-economic status, age, gender and more.

## STEPS TO ACHIEVE THE TASK
1. Import necessary libraries
- Pandas : For data manipulation
- Numpy: For handling numerical data.
- Seaborn: To load the Titanic dataset
- Streamlit: To create a web application
- Matplotlib: For data visualization
- Scikit-learn (sklearn): For machine learning (train-test split, logistic regression, and accuracy scoring).

2. Load Titanic Dataset
3. Data Preprocessing
   - Display the first few rows of the dataset
   - Check for missing values
   - Fill missing 'age' with the median value
   - Fill missing 'embarked' with the most frequent port 'S'
   - Removes irrelevant columns (deck, alive, class, who, embark_town)
   - Convert categorical columns (sex, alone, adult-male and embarked) to numeric

4. Defines Features (X) and Target (y)
   - X (Independent Variables): Passenger information.
   - y (Dependent Variable): Whether the passenger survived (1) or not (0)

5. Trains a Logistic Regression Model
   - Splits the data 80% for training and 20% for testing.
   - Fits a Logistic Regression model (iterating up to 1000 times to ensure convergence).
  
6. Creates a Streamlit Web UI
   - Displays a title and description
   - Accept input from users
   - Convert inputs to numerical values

    #### What is Streamlit?
   Streamlit is a Python library used to create interactive web apps for data science and machine learning with minimal code. It allows users to interact with models, visualize data, and     make predictions directly from a web browser.

   #### How the Streamlit UI Works
   - st.title():  Displays the app title
   - st.write():  Adds descriptive text to guide users.
   - st.selectbox(): Dropdown menu (e.g., Passenger Class selection)
   - st.radio(): Radio buttons (Gender selection).
   - st.slider(): Slider for numerical input (Age, Siblings Aboard).
   - st.number_input(): Numeric input (Fare paid).
   - st.button("Predict Survival"): Adds a button to trigger prediction.
  
   #### How to Copy Paste Emojis for Your Code
   - Copy from an Emoji Website: Emojipedia or Get Emoji 
   - Click Copy
   - Paste it into your Python code

8. Predict survival
   When the "Predict Survival" button is clicked
   - The app formats user input into an array.
   - The model makes a prediction (1 = Survived, 0 = Not Survived).

9. Show Model Accuracy
   - Computes and displays the accuracy of the logistic regression model.
     
     #### What is accuracy_score?
     The accuracy_score function from scikit-learn measures how well a classification model performs. It calculates the proportion of correctly predicted labels to the total number of         predictions.

     #### Explanation on usage
     - y_test: The actual survival outcomes from the test set.
     - model.predict(X_test): The predicted survival outcomes by the Logistic Regression model.
     - accuracy_score(y_test, model.predict(X_test)): Compares the actual vs. predicted values and returns the accuracy.
    
     #### What does Accuracy Mean?
     - 80-100%: Model is performing well.
     - 50-70%: Model is okay but could improve
     - Below 50%: Model is worse

10. Visualize Feature Importance
    - Shows which features (age, fare, class, etc.) were most important in predicting survival.
   
## STEPS TO RUN STREAM LIT APP
- Open your Anaconda Prompt if you have Anaconda installed on your system or Command Prompt
- Navigate to your script’s folder
- Run the Streamlit app: after locating the folder, type (streamlit run titanicc_app.py).
