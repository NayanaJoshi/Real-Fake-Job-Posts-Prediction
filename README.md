# Leveraging NLP to Detect Fake Job Postings

## Executive Summary
Fake job ads can deceive people and harm their trust in online platforms. This project, "Leveraging NLP to Detect Fake Job Postings," focuses on building a Machine Learning Model that identifies fake job postings by analyzing how they’re written. It aims to enhance safety and trust by spotting suspicious posts effectively.

---

## Problem Statement
### Background
In today’s digital age, online job platforms have become a primary source for job seekers to find employment opportunities. However, fake job postings have emerged as a significant issue. These frauds mislead job seekers, waste their time, steal personal information, and may even cause financial loss.

### Objective
The objective is to create a reliable model that can identify and prevent fake job postings before they reach job seekers.

### Scope
The model will analyze job postings for unreal patterns and inconsistencies, identifying fake jobs to prevent them from reaching job seekers. By protecting users from fake jobs, the solution will benefit both individuals and job platforms, ensuring a safer and more trustworthy online job search experience.

---

## Data Sources
- **Primary Data:** N/A
- **Secondary Data:** Real/Fake job postings dataset from Kaggle

---

## Methodology
### Data Collection
- Collecting and importing data from Kaggle datasets.

### Data Preparation
- Cleaning data
- Handling missing data
- Dropping unnecessary columns
- Implementing basic NLP preprocessing techniques such as tokenization, vectorization, etc.

### Analysis Techniques
- Exploring different classification models such as:
  - Support Vector Classifier (SVC)
  - Random Forest
  - Logistic Regression
  - SGDClassifier

### Tools
- **Excel:** For preliminary data analysis
- **Python:** For modeling, using libraries such as pandas, scikit-learn, and NLP libraries

---

## Expected Outcomes
- An ML model capable of distinguishing between real and fake job postings.
- A solution that can be scaled for future inventory planning.
- A safer and more trustworthy online job market for both job seekers and employers.

---

## Risks and Challenges
- Scammers constantly evolving tactics to make fake job postings appear legitimate.
- The possibility of false positives, where genuine job postings are flagged as fake, which could negatively impact employers.
- Limited availability of high-quality data to train the model.
- Ensuring the model performs well across different types of job postings and languages.

---

## Conclusion
Fake job ads pose a significant threat to job seekers and online job platforms. This project aims to develop a robust model to identify and stop these fake postings, creating a safer and more trustworthy environment for online job searching. By protecting individuals from scams and ensuring the reliability of job platforms, this project can significantly contribute to the integrity of the online job market.

---

## How to Use This Repository
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NayanaJoshi/Real-Fake-Job-Posts-Detection.git
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then install required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project:**
   Follow the steps in the provided Jupyter Notebook to preprocess data, train the model, and evaluate results.

4. **Contribute:**
   Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For any questions or feedback, feel free to reach out:
- **Email:** joshinayana206@gamil.com
- **GitHub:** NayanaJoshi(https://github.com/NayanaJoshi/Real-Fake-Job-Posts-Prediction)

