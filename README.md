# soma-gen
# ReneWind: Wind Turbine Failure Prediction (Neural Networks)
## 📌 Project Overview
Unscheduled downtime in wind energy infrastructure leads to catastrophic operational costs and revenue loss. **ReneWind** is a deep learning-driven predictive maintenance system designed to parse complex sensor telemetry logs and accurately classify wind turbine generator failures before they occur. 

The primary business and technical goal of this project is **proactive maintenance optimization**, intentionally tuning the neural network architecture to minimize false negatives so that critical components are flagged for repair prior to complete mechanical breakdown.

## 🛠️ Key Technical Implementations
* **Deep Learning Classification:** Engineered and optimized multi-layer feedforward Neural Networks, experimenting with custom activation functions and dropout layers to parse highly non-linear sensor relationships.
* **Exploratory Data Analysis (EDA):** Executed intensive preprocessing, anomaly detection, and correlation analysis to map real-time sensor telemetry variances directly to specific mechanical failure modes.
* **Cost-Sensitive Evaluation:** Implemented custom loss functions and focused model evaluation heavily on Recall metrics to guarantee that high-risk failure modes are rarely missed by the network.

## 📊 Model Evaluation & Metrics
Because a missed failure (False Negative) is vastly more expensive than a false alarm (False Positive), the model was optimized to maximize Recall while maintaining strong overall precision.

| Model Architecture | Precision | Recall (Failures) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Neural Network | 0.76 | 0.81 | 0.78 | 0.85 |
| **Optimized NN (Custom Activation + Dropout)** | **0.84** | **0.94** | **0.89** | **0.95** |

## 2. EasyVisa: Predictive Modeling Framework

```markdown
# EasyVisa: Predictive Modeling Framework (Advanced Machine Learning)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![ML Libraries](https://img.shields.io/badge/Libraries-Scikit--Learn%20%2F%20XGBoost-green)
![Data Balancing](https://img.shields.io/badge/Techniques-SMOTE%20%2F%20Undersampling-blueviolet)

## 📌 Project Overview
The visa application process often suffers from severe processing bottlenecks due to manual profile evaluations. **EasyVisa** is an automated predictive modeling framework built to streamline and accelerate applicant screening. By assessing historical application records, the framework provides logical, data-backed candidate profiles to guide processing compliance and flag high-risk applications.

## 🛠️ Key Technical Implementations
* **Ensemble Modeling Benchmark:** Built, evaluated, and benchmarked over **15 distinct classification models** including Decision Trees, Random Forests, Bagging, Boosting (XGBoost, LightGBM), and Stacking Classifiers.
* **Imbalanced Data Handling:** Remedied severe class imbalance (where approved visas vastly outnumbered denied visas) utilizing rigorous oversampling (SMOTE) and undersampling techniques to stabilize performance metrics.
* **Systematic Hyperparameter Tuning:** Utilized `GridSearchCV` and `RandomizedSearchCV` to fine-tune tree depth, learning rates, and estimator counts, significantly boosting model generalization on unseen data.

## 📊 Model Comparison Ledger
Below is a summary of the top-performing architectures from the 15+ model benchmark:

| Model Type | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Decision Tree (Baseline) | 79.2% | 0.74 | 0.71 | 0.72 |

## 3. Personal Loan Campaign: Customer Target System

```markdown
# Personal Loan Campaign: Customer Target System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Algorithm-Decision%20Trees-yellowgreen)
![Optimization](https://img.shields.io/badge/Optimization-Pre%2FPost%20Pruning-brightgreen)

## 📌 Project Overview
Mass marketing campaigns are highly inefficient and costly. This project develops a high-conversion **Propensity Model** tailored for a liability bank to isolate existing customers with the highest statistical likelihood of purchasing a personal loan. By converting banking demographics into actionable decision rules, the institution can optimize its marketing spend and boost conversion rates.

## 🛠️ Key Technical Implementations
* **Tree Pruning & Generalization:** Implemented meticulous pre-pruning (setting `max_depth` and `min_samples_split`) and post-pruning (cost-complexity pruning $\alpha$) to prevent model overfitting while maintaining highly interpretable business rules for stakeholders.
* **End-to-End Feature Engineering:** Managed the complete data lifecycle, including missing value imputation, robust outlier mitigation, categorical encoding, and feature scaling.
* **Interpretability Mapping:** Extracted and visualized the optimized decision paths to allow marketing teams to see exactly *why* a customer profile was flagged as a high-conversion target.

## 📈 Business Results & Impact
* **Overfitting Elimination:** Pruning successfully closed the training/testing accuracy gap from an overfitted 10% variance down to **<1.5% variance**.
* **Target Optimization:** The final decision rules allowed the bank to isolate a subgroup of customers that yielded a **3x higher conversion rate** than random baseline targeting.


| Random Forest (Balanced) | 86.5% | 0.83 | 0.81 | 0.82 |
| **XGBoost + Hyperparameter Tuning** | **91.4%** | **0.89** | **0.88** | **0.88** |
