# 7005SCN INDIVIDUAL RESEARCH PROJECT

## CW1 Project Proposal (Proforma)

**Word Count:** 1000 words ±10%

| Field | Details |
| --- | --- |
| **Student Name** | AIYERVBOSA SHALOM ANTHONY |
| **Student ID** | 15621427 |
| **Course** | |
| **Supervisor** | ALISON HALFORD |

### Ethics Application Status

Tick ONE option:

- ☒ **Approved** — Ethics ID: P194689
- ☐ Submitted, under review
- ☐ Not yet submitted

---

## Research Question/Problem Statement

Clearly define your research problem or question (or hypothesis), situating it within its relevant domain or context. Explain why this problem is important, what gap or issue it addresses and why it is worthy of investigation.

Financial fraud is a growing challenge in the digital economy due to increased online banking, mobile payments, e-wallets and fintech services. Although these technologies improve access and efficiency, they also create opportunities for identity theft, unauthorised transactions and cyber-enabled fraud. Traditional rule-based systems often fail to detect new fraud patterns because they depend on fixed rules and thresholds, leading to missed fraud cases or false alarms.

Machine learning offers a stronger solution by learning from historical transaction data and identifying suspicious patterns. However, many existing systems are complex, costly and difficult to apply in academic settings. This research develops a lightweight machine learning-based fraud detection and financial risk monitoring prototype that predicts fraud, assigns risk levels and visualises alerts.

Key stakeholders include financial institutions, fintech firms, fraud analysts, risk teams, researchers and developers who need accurate, interpretable and affordable tools to support fraud prevention and decision-making.

---

## Intended Users/Stakeholders and Needs

Identify the key users, stakeholders or organisations who will benefit from your project. Explain their needs, challenges and describe how your proposed research aims to address or respond to these needs.

The main stakeholders are financial institutions, fintech organisations, fraud analysts, risk management teams, researchers and software developers. They need reliable tools to detect fraud, reduce financial losses and improve transaction monitoring. Financial institutions require fast and accurate detection with fewer false alarms, while fraud analysts need clear explanations for flagged transactions.

Fintech firms need scalable and affordable solutions that respond to changing fraud patterns. Researchers and developers benefit from practical machine learning frameworks. This project meets these needs by developing a fraud detection and risk monitoring prototype that predicts fraud, assigns risk levels and supports better decision-making.

---

## Project Scope, Deliverables and Expected Outcomes

Outline the scope and boundaries of your project, including what will and will not be included. Describe the artefact, system, model or analysis you intend to produce, the key deliverables, and the expected outcomes, making clear how these align with and address the research problem.

The scope of this project is limited to the development and evaluation of machine learning models for fraud detection using a publicly available financial transaction dataset. The study focuses on identifying fraudulent transactions and monitoring transaction risk levels through a prototype system.

### The project will include:

- Literature review on fraud detection and financial risk monitoring.
- Data preprocessing and exploratory analysis.
- Development and training of machine learning models.
- Comparative performance evaluation of selected algorithms.
- Development of a prototype monitoring dashboard.
- Analysis and interpretation of results.

### The project will not include:

- Real-time deployment in a live banking environment.
- Integration with commercial banking systems.
- Processing of real customer financial data.
- Enterprise-scale cybersecurity infrastructure.

### Key deliverables include:

- A comparative analysis of machine learning fraud detection models.
- A trained fraud prediction model.
- A prototype fraud detection and financial risk monitoring system.
- A dissertation documenting methodology, findings, and recommendations.

### Expected outcomes include:

Identifying the most effective machine learning algorithm for fraud detection, improving understanding of fraud prediction techniques, and demonstrating how machine learning can support financial risk monitoring through visual analytics.

---

## Research Methodology

Explain how your research will be conducted by outlining your overall approach (e.g., experimental, case study, design-based, analytical). Describe the methods, techniques, tools, technologies and data sources you will use, including how data will be collected and analysed where relevant. Provide justification for your methodological choices.

This study will adopt a quantitative experimental research methodology. Secondary data will be obtained from the CiferAI/Cifer-Fraud-Detection-Dataset-AF available on Hugging Face. The dataset contains anonymised financial transaction records labelled as either fraudulent or legitimate.

The research process will consist of several stages. First, data preprocessing will be conducted through cleaning, handling missing values, feature engineering, feature scaling, and addressing class imbalance using suitable techniques such as oversampling or undersampling. Exploratory data analysis will then be performed to identify trends, patterns, and relationships within the dataset.

Several machine learning algorithms will be implemented and compared, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The models will be developed using Python and relevant libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

Model performance will be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Following evaluation, the best-performing model will be integrated into a prototype fraud detection and financial risk monitoring system. The prototype will visualise fraud predictions, transaction risk levels, and fraud alerts using dashboard components.

This methodology is appropriate because it enables objective comparison of machine learning techniques while supporting the development of a practical proof-of-concept system that demonstrates real-world applicability.

---

## Initial Literature Review

Provide a concise but critical review of 3–5 academic sources. Summarise key findings and approaches, compare and synthesise perspectives, identify gaps or limitations in the existing literature and explain how your proposed project will address these gaps.

Machine learning has become a widely adopted approach for fraud detection due to its ability to identify complex patterns within financial transaction data. Talukder, Khalid and Uddin (2024) developed a multistage ensemble machine learning model and demonstrated improved fraud detection performance compared to single-model approaches. However, the framework required substantial computational resources.

Farouk et al. (2024) investigated supervised machine learning techniques for online payment fraud detection and reported high detection accuracy through ensemble learning approaches. Nevertheless, the study highlighted challenges associated with real-time deployment and operational scalability.

Hu (2025) proposed an improved fraud detection framework combining Random Forest and Gradient Boosting Machine algorithms. The study achieved high predictive accuracy and robustness but involved increased model complexity and training requirements.

Almalki and Masud (2025) explored explainable artificial intelligence using XGBoost combined with SHAP and LIME techniques. Their findings demonstrated that explainability can improve transparency and trust in fraud detection systems while maintaining strong predictive performance. However, computational overhead remained a challenge.

Çavdar and Bozanta (2026) found that explainable machine learning improves credit card fraud detection by combining accuracy with interpretability, although deployment remains complex. In summary, machine learning enhances fraud detection, but issues such as cost, scalability and usability remain. This project addresses these gaps through a lightweight, interpretable prototype.

---

## Project Planning and Management

Provide a structured and feasible plan explaining how your project will be carried out over time. This should include a clear timeline of key stages, tasks and milestones, demonstrating sequencing and dependencies. You should also explain the feasibility of the project in terms of time, resources and scope and identify potential risks along with appropriate mitigation strategies.

The project will be completed within 10 weeks using a structured and realistic schedule. Weeks 1 and 2 will focus on reviewing relevant literature and refining the research proposal. This stage will help establish the theoretical background of the study, identify gaps in fraud detection research and clarify the aims, objectives and research questions.

Week 3 will involve dataset identification, collection and preparation. A publicly available fraud detection dataset will be selected, and the data will be checked for suitability. Week 4 will focus on preprocessing activities, including data cleaning, handling missing values, feature selection, scaling and preparing the dataset for machine learning analysis.

Week 5 will be used for Exploratory Data Analysis (EDA). This will involve examining fraud patterns, transaction trends, relationships between variables and the level of class imbalance in the dataset. Weeks 6 and 7 will focus on developing and training selected machine learning models, including Logistic Regression, Decision Tree, Random Forest and XGBoost.

Week 8 will be dedicated to model evaluation and comparison using accuracy, precision, recall, F1-score and ROC-AUC. This will help identify the most effective model for detecting fraudulent transactions. Week 9 will focus on developing a simple prototype fraud detection and financial risk monitoring system to display fraud predictions, risk scores and alerts. The prototype will also be tested for functionality, reliability and usability.

Week 10 will be used for dissertation writing, review, proofreading, refinement and final submission. The methodology, findings, analysis, conclusions and recommendations will be clearly documented.

The project is feasible within 10 weeks because it uses publicly available datasets, open-source machine learning tools and a prototype-based approach. Key risks include poor dataset quality, class imbalance, low model accuracy, technical issues and time constraints. These will be managed through preprocessing, SMOTE, model comparison, parameter tuning, testing and regular progress monitoring.

---

## References (APA Style)

List all references using APA format.

Almarshad, F. A., Zakariah, M., & Gashgari, G. A. (2025). Risk-adaptive Bayesian ensemble model for fraud detection. *Scientific Reports*, *15*(36796).

Almalki, F., & Masud, M. (2025). Financial fraud detection using explainable AI and stacking ensemble methods. *arXiv*.

Çavdar, E. Z., & Bozanta, A. (2026). AI-based credit card fraud detection: A machine learning approach with model explainability on real-world data. *Knowledge and Information Systems*.

Hu, T. (2025). Financial fraud detection system based on improved random forest and gradient boosting machine. *arXiv*.

Jin, J., & Zhang, Y. (2025). The analysis of fraud detection in financial market under machine learning. *Scientific Reports*, *15*(29959).

Narasapuram, N. K., et al. (2026). Explainable artificial intelligence models for detecting suspicious bank transactions. *International Journal of Machine Learning and Cybernetics*, *17*(111).

Sun, W., Qi, Z., & Shen, Q. (2025). High-recall deep learning: GRU approach to bank fraud detection. *SSRN*.

Talukder, M. A., Khalid, M., & Uddin, M. A. (2024). An integrated multistage ensemble machine learning model for fraudulent transaction detection. *Journal of Big Data*, *11*(168).

Zioviris, G., Kolomvatsos, K., & Stamoulis, G. (2024). An intelligent sequential fraud detection model based on deep learning. *Journal of Supercomputing*, *80*, 14824–14847.

Zavvar, M., et al. (2025). A hybrid deep learning framework using synthetic oversampling and attention mechanisms for credit card fraud detection. *Journal of Big Data*.

---

## AI Usage Declaration

Explain if and how AI tools were used.

Artificial Intelligence tools, including ChatGPT, were used to support the development of this proposal. AI was utilised to assist with academic writing, language refinement, proposal structuring, literature synthesis, and formatting. All content generated through AI assistance was critically reviewed, edited, and validated by the researcher to ensure academic integrity, accuracy, and alignment with the project requirements. The final submission represents the student's own work and understanding of the research topic.
