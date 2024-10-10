# BreastCancerAI-

### Project: BreastCancerAI

**Overview:**  
The BreastCancerAI project aimed to develop a robust predictive model for breast cancer diagnosis using various machine learning techniques, including Random Forest, Naive Bayes, and Neural Networks. The primary goal was to identify malignancies in mammography images and clinical data accurately, facilitating earlier detection and improved treatment outcomes.

**Technologies Used:**  
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy  
- **Framework:** Keras for building and training neural networks

**Project Phases:**

1. **Data Collection:**  
   - The project utilized a publicly available breast cancer dataset containing features related to tumor characteristics and diagnostic labels.

2. **Data Preprocessing:**  
   - Data was cleaned and preprocessed, including normalization and encoding categorical variables. Missing values were handled, ensuring high-quality input for the models.

3. **Model Development:**
   - **Random Forest Classifier:** A Random Forest model was implemented to leverage ensemble learning. It combined multiple decision trees to improve accuracy and reduce overfitting, achieving a solid baseline performance.
   - **Naive Bayes Classifier:** This algorithm was used for its simplicity and effectiveness in handling classification tasks, particularly for high-dimensional datasets.
   - **Neural Network:** A deep learning model was constructed using Keras and TensorFlow, featuring multiple layers to capture complex patterns in the data. The architecture included input, hidden, and output layers, with activation functions such as ReLU and sigmoid.

4. **Model Training and Evaluation:**  
   - Each model was trained on a training dataset and evaluated using a separate validation set. Metrics such as accuracy, precision, recall, and F1 score were computed to assess performance.
   - The neural network model outperformed the other algorithms, achieving an accuracy of 95%.

5. **Results and Insights:**  
   - The neural network model demonstrated superior performance in diagnosing breast cancer compared to Random Forest and Naive Bayes. 
   - The insights gained from model predictions can guide healthcare professionals in making informed decisions regarding patient diagnosis and treatment.

6. **Conclusion:**  
   - The BreastCancerAI project successfully showcased the potential of machine learning in healthcare. The neural network model not only achieved high accuracy but also contributed to earlier detection of breast cancer, highlighting the importance of leveraging technology in medical diagnostics. Future work can focus on further refining the model and exploring additional features or data sources to enhance predictive power. 

**Impact:**  
This project emphasizes the transformative role of data science in improving healthcare outcomes, demonstrating how advanced algorithms can support early diagnosis and better patient management.
