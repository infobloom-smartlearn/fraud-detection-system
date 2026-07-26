# Chapter Four: Artefact Design and Development

## 4.1 Artefact Requirements and Specifications

The artefact was specified as a lightweight fraud detection and financial risk monitoring prototype capable of classifying synthetic financial transactions, comparing supervised learning algorithms, and presenting results through an accessible web interface. Functional requirements were derived from the project proposal and stakeholder needs identified in Chapter One. The system must ingest transaction attributes—including simulation step, transaction type, amount, origin and destination balances, account identifiers, and rule-engine flags—and return a binary fraud classification, a fraud probability score, and a categorical risk level (Low, Medium, or High). Non-functional requirements included reproducibility of the training pipeline, modularity between preprocessing and inference components, and deployment feasibility within an academic budget using open-source tools.

Performance requirements specified comparative evaluation of four algorithms—Logistic Regression, Decision Tree, Random Forest, and XGBoost—using accuracy, precision, recall, F1-score, and ROC-AUC on a held-out test set. The best-performing model was required for production integration. Usability requirements demanded a dashboard summarising model metrics and a form-based interface for single-transaction analysis. Scope boundaries excluded live banking integration, real customer data processing, and enterprise infrastructure, ensuring the artefact remained an ethical proof-of-concept aligned with approved ethics application P194689 and institutional research governance.

---

## 4.2 System or Solution Architecture

The solution follows a three-tier architecture comprising a data layer, a machine learning layer, and a presentation layer, as illustrated in Figure 4.1.

**Figure 4.1: System architecture of the fraud detection prototype**

```
┌─────────────────────────────────────────────────────────────┐
│                  Presentation Layer (Flask)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Overview    │  │  Transaction │  │  REST API        │ │
│  │  Dashboard   │  │  Analysis UI │  │  /api/predict    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │                 │                   │
┌─────────▼─────────────────▼───────────────────▼─────────────┐
│              Inference Pipeline (FraudDetectionPipeline)     │
│  Feature Engineering → Account Registry Lookup → Prediction  │
└─────────┬───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│                    Machine Learning Layer                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Trained     │  │ Threshold    │  │ StandardScaler      │ │
│  │ Classifier  │  │ Classifier   │  │ (where applicable)  │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└─────────┬───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│                       Data Layer                               │
│  Cifer Dataset → Cleaning → Feature Engineering → 50/50 Split│
│  Account Legitimacy Registry (train corpus)                   │
└───────────────────────────────────────────────────────────────┘
```

Raw transaction data from the Cifer-Fraud-Detection-Dataset-AF is processed offline to produce engineered features and a balanced evaluation dataset. An Account Legitimacy Registry maps sender and receiver identifiers to historical legitimate activity across the training corpus. At runtime, the Flask application loads a serialised FraudDetectionPipeline object containing the deployed Decision Tree model, the registry, feature column definitions, and a tuned classification threshold. User requests flow through template-rendered pages or JSON API endpoints, ensuring separation between research experimentation (offline training scripts) and operational inference (web service via Gunicorn on Render).

The presentation layer exposes three interaction modes aligned with stakeholder needs: a research overview for supervisory and examiner review, an analyst-facing transaction form for ad hoc classification, and a REST endpoint for programmatic integration testing. The machine learning layer encapsulates preprocessing logic so that training-time and inference-time transformations remain consistent, reducing deployment skew. The data layer supports multi-part CSV ingestion, enabling scalability from single-part experimentation to full-corpus processing without architectural redesign. Offline and online components share the FraudDetectionPipeline abstraction to prevent feature drift between experimentation and deployment. Security considerations at this stage focused on environment-variable management for secret keys and exclusion of real personal data, consistent with the synthetic dataset constraint established in the ethics approval.

---

## 4.3 Design Decisions

Several design decisions shaped the artefact. First, a **50/50 balanced sampling strategy** was adopted because the raw dataset is highly imbalanced (fraud constitutes approximately 0.13% of transactions). Balancing legitimate and fraudulent records enabled meaningful comparison of classifiers without majority-class dominance, while acknowledging that production fraud rates differ—a limitation discussed in Chapter Six.

Second, **account legitimacy features** (`orig_in_global_legit`, `dest_in_global_legit`) were engineered after exploratory analysis revealed that fraudulent accounts rarely appear in legitimate transaction histories within the corpus. This decision improved separability substantially compared with transaction-only attributes alone, where models initially achieved near-random accuracy (~50%). The registry was built from all legitimate activity in the training parts before splitting, ensuring consistent inference for unseen accounts at deployment.

Third, the **Decision Tree** was selected as the deployed model based on highest test accuracy (91.94%), F1-score (0.924), and ROC-AUC (0.987), prioritising interpretability over ensemble complexity. Fourth, a **ThresholdClassifier** wrapper was implemented to optimise decision boundaries on a validation subset rather than relying on the default 0.5 probability cut-off, improving metric alignment with project objectives.

Finally, the web interface was designed with an academic presentation style—research overview, methodology summary, evaluation table, and transaction analysis form—rather than a commercial dashboard, reflecting the artefact's role as a research demonstrator. Highest test accuracy was adopted as the definitive model selection criterion after comparative tuning, and all experiments were documented in version-controlled CSV and JSON artefacts.

---

## 4.4 Tools, Technologies and Techniques

Development was conducted in Python 3.12 using an experimental pipeline suited to tabular classification. **Pandas** and **NumPy** supported data loading, cleaning, and numerical operations across multi-part CSV files totalling approximately 21 million records. **Scikit-learn** provided preprocessing (`StandardScaler`, `train_test_split`, `StratifiedKFold`, `RandomizedSearchCV`), baseline classifiers, and evaluation metrics. **XGBoost** extended gradient-boosted tree capability within the comparative study. **Imbalanced-learn** was available for resampling during initial preprocessing experiments.

Feature engineering techniques included balance-change computation, logarithmic amount transformation, transaction-type one-hot encoding, hour-of-day and simulation-day extraction, and account legitimacy indicators. Model tuning employed randomised hyperparameter search with accuracy-focused scoring and validation-set threshold selection.

The prototype interface was implemented with Flask, Jinja2 templates, and custom CSS styled for formal academic presentation. Production deployment used Gunicorn as the WSGI server and Render for cloud hosting, with render.yaml defining build and start commands. Joblib serialised trained models and the inference pipeline. Matplotlib and Seaborn generated confusion matrices and ROC curves during evaluation. Structured project directories (src/, app/, models/, outputs/) and documented scripts (train_fast.py, preprocess_and_eda.py) ensured that each development stage could be repeated independently by examiners or future researchers.

---

## 4.5 Implementation and Development Process

Implementation proceeded through six staged activities aligned with the research methodology.

**Stage 1: Data acquisition and exploration.** Fourteen dataset parts were downloaded from Hugging Face. Exploratory analysis on Part 1 confirmed severe class imbalance, transaction-type distributions, and weak linear separability using transaction attributes alone.

**Stage 2: Preprocessing and feature engineering.** A reusable module (`dataset_utils.py`) implemented cleaning rules (duplicate removal, invalid amount/step filtering), PaySim-inspired derived features, and account legitimacy enrichment. A balanced dataset of 3,968 records (1,984 fraud; 1,984 legitimate) was constructed from Part 1 for final model training, with registry coverage of approximately 1.48 million legitimate origin accounts.

**Stage 3: Model training and tuning.** Four classifiers were trained via `train_fast.py` with candidate hyperparameter configurations per algorithm. Each model was evaluated on a stratified 80/20 train-test split (with an internal 15% validation fold for threshold tuning). Table 4.1 presents final test-set results.

**Table 4.1: Comparative model performance on the held-out test set**

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 91.94% | 87.58% | 97.73% | 0.924 | 0.987 |
| Logistic Regression | 84.38% | 76.20% | 100.00% | 0.865 | 1.000 |
| XGBoost | 82.24% | 100.00% | 64.48% | 0.784 | 1.000 |
| Random Forest | 79.22% | 70.64% | 100.00% | 0.828 | 1.000 |

**Stage 4: Model selection and serialisation.** The Decision Tree was exported as best_model.joblib and embedded within FraudDetectionPipeline for unified preprocessing and prediction. The pipeline bundles the classifier, feature column list, account registry, and tuned threshold so that a single load operation prepares the application for inference. **Stage 5: Web application development.** Flask routes were created for the research overview (/), transaction analysis (/predict), health monitoring (/health), and JSON inference (/api/predict). The interface was iteratively redesigned to meet dissertation presentation standards, incorporating system status indicators, methodology panels, and a comparative metrics table. Form validation ensures required transaction fields are present before prediction, and results are rendered with fraud probability, binary classification, and risk band. **Stage 6: Deployment configuration.** Production dependencies were isolated in requirements-prod.txt to reduce memory footprint on Render. The WSGI entry point (wsgi.py) and Render blueprint were configured, with health-check endpoints validated locally before cloud release. These stages completed the artefact from trained model to deployable web service. Figure 4.2 presents the research overview dashboard; Figure 4.3 presents the transaction analysis interface (diagrams to be inserted).

**Figure 4.2: Research overview dashboard — model evaluation summary and system status**

*[Insert screenshot: Overview page showing hero banner, system status panel, research scope cards, and model comparison table with Decision Tree marked as Deployed]*

**Figure 4.3: Transaction analysis user interface**

*[Insert UI diagram/screenshot provided by student: transaction input form and prediction result panel showing fraud classification, probability, and risk level]*

---

## 4.6 Testing and Technical Validation

Testing combined quantitative model evaluation with functional verification of the deployed artefact. **Model validation** used stratified splitting to preserve class proportions, reporting metrics on data not seen during training. The Decision Tree achieved the strongest overall balance of accuracy and recall, correctly identifying 97.73% of fraud cases while maintaining 87.58% precision. ROC-AUC of 0.987 indicates strong ranking ability across classification thresholds.

**Functional testing** verified the Flask application using `app.test_client()`, confirming HTTP 200 responses for the overview and prediction routes. The `/health` endpoint returned `{"status": "ok", "model_loaded": true}`, validating pipeline initialisation. Manual test cases included legitimate-account transactions (low fraud probability) and unknown-account transactions with fraud-like attribute patterns (elevated probability).

**Integration testing** confirmed end-to-end flow from form submission through feature engineering, registry lookup, model inference, and risk-level assignment. **Deployment testing** replicated the Render production command locally (gunicorn wsgi:app), ensuring compatibility before cloud release. **Regression testing** after UI redesign verified that model loading and metric display remained functional. API responses were validated. Limitations include evaluation on balanced rather than naturally imbalanced data and single-part training for the final reported metrics; these are addressed analytically in Chapter Six.

---

## 4.7 Artefact Justification and Chapter Summary

The artefact satisfies the project aim by demonstrating that a lightweight, open-source fraud detection prototype can achieve strong classification performance and support analyst-facing monitoring. Account legitimacy features and careful threshold tuning transformed initially weak models into viable detectors. The Decision Tree offered the best accuracy–interpretability trade-off for deployment. The modular pipeline, documented training scripts, and web interface together provide a reproducible research artefact suitable for examiner demonstration and future extension work. Chapter Four documented requirements, architecture, design rationale, technologies, implementation stages, and validation, establishing the technical foundation evaluated in subsequent chapters.

---

# Chapter Five: Project Management

## 5.1 Project Planning Approach

The project followed a structured plan derived from the approved CW1 proposal, dividing work into literature review, data preparation, model development, artefact construction, evaluation, and dissertation writing. A work-breakdown structure aligned each activity with research objectives and deliverable deadlines. Weekly progress was tracked against the academic calendar using a Gantt chart (Appendix reference), with supervisor meetings providing formative feedback. An iterative development approach was adopted for the artefact, enabling justified refinement when initial results proved inadequate. Time was allocated in phases: research and ethics first, then data and modelling, then artefact integration and writing. Reviews ensured alignment with module deadlines.

## 5.2 Timeline, Milestones and Deliverables

Key milestones included: ethics approval (P194689); completion of Chapter Two literature review; dataset acquisition and preprocessing; baseline model training; feature enhancement and retraining; Flask prototype development; UI redesign for academic presentation; Render deployment setup; and final dissertation submission. Major deliverables comprised the comparative model analysis (model_comparison_latest.csv), trained pipeline artefacts, the web application, evaluation outputs, and written chapters. The most time-intensive milestone was data processing and model iteration, originally estimated at two weeks but extended to three due to class imbalance and feature discovery. Interim deliverables such as EDA plots and baseline metrics were scheduled before web development to ensure evidence supported design decisions. Each milestone was mapped to a calendar week in the project Gantt chart.

## 5.3 Risk Management

A risk register identified potential threats and mitigations. Technical risks included poor model performance (mitigated through additional feature engineering and algorithm comparison), large dataset processing times (mitigated via part-based loading and quick-training scripts), and deployment memory limits on Render free tier (mitigated by single-worker Gunicorn configuration). Schedule risks included underestimating preprocessing complexity (mitigated by reprioritising scope to Part 1 for final metrics). Ethical risks were controlled through exclusive use of anonymised synthetic data and approved ethics clearance. Risks were reviewed at supervisor meetings and re-scored when scope or findings changed. No critical risks materialised without a documented response.

## 5.4 Challenges and Project Changes

Significant challenges emerged during development. Initial models achieved near-random accuracy because transaction-level features alone lacked discriminative power in balanced samples. Extensive analysis identified account legitimacy indicators as the critical signal, requiring architectural changes to the preprocessing pipeline. File permission issues on Windows occasionally blocked CSV overwrites during training, resolved by writing to alternate output filenames. The best-model selection criterion was revised from an accuracy-band filter to highest test accuracy after evaluation showed Decision Tree superiority. The user interface was redesigned to meet dissertation presentation standards, and Render deployment was added to demonstrate applicability beyond local execution.

## 5.5 Reflection on Project Management Effectiveness

Overall, planning proved effective in maintaining alignment with objectives, though technical discovery required justified schedule adjustment. Early prototyping exposed weaknesses before deployment investment, validating the iterative approach. Supervisor engagement and clear milestone tracking reduced scope drift. The project delivered core deliverables within the academic timeframe. Future projects would benefit from earlier exploratory data analysis milestones and explicit buffers for deployment configuration within the academic schedule. Change logs supported accountability when scope shifted.

---

# Chapter Six: Evaluation and Discussion

## 6.1 Introduction to the Evaluation

This chapter evaluates the completed artefact and experimental findings against the original research aim: to develop and assess a lightweight machine learning-based fraud detection prototype using publicly available transaction data. Evaluation considers quantitative model metrics, functional capabilities of the web application, alignment with stated objectives, and comparison with prior literature reviewed in Chapter Two. The discussion interprets rather than restates results, acknowledging both achievements and methodological constraints that bound the claims presented in the sections below.

## 6.2 Presentation of Results or Findings

Experimental evaluation compared four supervised classifiers on a balanced test set of 794 transactions (20% hold-out from 3,968 records). The Decision Tree achieved the highest accuracy (91.94%), F1-score (0.924), and ROC-AUC (0.987), with recall of 97.73%. Logistic Regression reached 84.38% accuracy with perfect recall (100%) but lower precision (76.20%), suggesting a tendency toward false positives. XGBoost exhibited the inverse profile: perfect precision (100%) but recall of only 64.48%, missing over one-third of fraud cases. Random Forest achieved 79.22% accuracy with full recall but the lowest precision among tree-based models (70.64%). These patterns align with the threshold-tuning and feature-engineering choices documented in Chapter Four.

The deployed prototype successfully classifies user-submitted transactions, returning fraud probability and risk levels through both interactive and API interfaces. Account legitimacy features proved decisive; without them, models performed at chance level on balanced data. The Account Legitimacy Registry, covering approximately 1.48 million legitimate accounts from Part 1, enabled generalisation to unseen identifiers at inference by flagging accounts with no legitimate history. Precision–recall trade-offs differed markedly across algorithms, indicating that model selection should reflect institutional priorities regarding false alarms versus missed fraud. These findings demonstrate that feature design, not algorithm complexity alone, drove performance gains in this study.

## 6.3 Evaluation Against the Research Objectives

The project objectives outlined in the proposal are assessed as follows. Objective 1: Conduct preprocessing and exploratory analysis — achieved through multi-stage cleaning, feature engineering, EDA outputs, and class balancing. Objective 2: Develop and compare ML models — achieved; four algorithms were trained, tuned, and compared using multiple metrics (Table 4.1). Objective 3: Identify the most effective algorithm — achieved; Decision Tree selected on highest accuracy and strong F1/ROC-AUC. Objective 4: Build a monitoring prototype — achieved via Flask dashboard and transaction analysis interface. Objective 5: Demonstrate practical applicability — partially achieved through Render deployment configuration; live hosting depends on repository publication but local and WSGI validation confirm readiness.

The research question—whether a lightweight ML prototype can effectively detect fraud and support risk monitoring—is answered affirmatively within the controlled experimental setting, with the caveat that performance metrics reflect a balanced evaluation design rather than natural class prevalence. Stakeholder needs identified in Chapter One for interpretability, affordability, and accessible monitoring are addressed through open-source tooling, Decision Tree transparency, and the professionally designed web interface. Gaps remain regarding real-time throughput and enterprise integration, explicitly excluded from scope but relevant for future industrial adoption. Overall, objectives were substantially met, with deployment representing the main area for post-submission completion. The comparative methodology satisfies the academic requirement to justify algorithm selection on evidence rather than default choices, and the monitoring interface demonstrates that research outputs can be communicated to non-technical stakeholders through structured visual presentation and accessible web design.

## 6.4 Comparison with Previous Studies

Chapter Two reviewed studies reporting high fraud detection accuracy using ensemble and boosting methods on imbalanced financial data. Talukder, Khalid and Uddin (2024) achieved strong results with multistage ensembles but required substantial compute. Farouk et al. (2024) emphasised ensemble learning for online payment fraud yet noted deployment scalability challenges. Hu (2025) combined Random Forest and GBM for robustness at increased complexity.

This project's Decision Tree (91.94% accuracy) compares favourably within the academic prototype context while maintaining interpretability—a factor emphasised by Bhattacharyya et al. (2011) regarding analyst trust. Unlike studies reporting near-perfect accuracy on severely imbalanced test sets (where majority-class classifiers inflate metrics), this evaluation used balanced hold-out data, producing more conservative but comparable cross-model discrimination. The Cifer dataset README cites higher benchmark accuracy; differences likely reflect evaluation protocol, feature sets, and registry construction. The present study contributes a transparent, reproducible pipeline with explicit account-level features and deployed inference—addressing the literature gap regarding accessible academic implementations identified in Chapter Two. The emphasis on lightweight deployment contrasts with compute-intensive ensemble pipelines cited in recent literature.

## 6.5 Strengths and Successful Outcomes

Key strengths include: a reproducible modular codebase; meaningful feature engineering grounded in data analysis; systematic four-model comparison; threshold optimisation; and integration into a professionally designed web interface suitable for research demonstration. The Decision Tree's interpretability supports explainability requirements for fraud analysts. Successful recovery from initial 50% accuracy demonstrates effective iterative research practice. Deployment artefacts extend academic work toward operational relevance without compromising ethics boundaries. The comparative metrics table embedded in the dashboard allows examiners to verify results without running training scripts. Together, these outcomes show that rigorous experimentation and practical delivery can coexist within an MSc research timeline.

## 6.6 Limitations, Reliability and Validity

Several limitations affect generalisability. Internal validity: final reported metrics derive from Part 1 (3,968 balanced records); although the pipeline supports all fourteen parts, broader multi-part evaluation was constrained by processing time. External validity: synthetic Cifer data mimics PaySim structure but may not fully represent live banking fraud patterns. Construct validity: balanced sampling improves classifier training but distorts accuracy relative to real-world base rates where fraud is rare. Reliability: stratified splitting and fixed random seeds support reproducibility; however, single train-test splits omit cross-validation reporting in final metrics. Registry construction from corpus-level legitimate accounts may inflate performance relative to strictly temporal train-only registries—a trade-off accepted for prototype demonstration but noted for production adaptation.

## 6.7 Overall Discussion

The evaluation confirms that supervised learning, combined with domain-informed account features, delivers effective fraud detection within a lightweight academic artefact. Trade-offs among precision, recall, and interpretability guide algorithm selection depending on institutional cost asymmetry. Production systems would require imbalanced evaluation and continuous retraining.

---

# Chapter Seven: Legal, Ethical and Social Considerations

## 7.1 Ethical Considerations

This project processed only anonymised synthetic financial data from the publicly available Cifer-Fraud-Detection-Dataset-AF, avoiding real customer records and thereby minimising direct harm to individuals. Ethics approval (P194689) was obtained prior to research activity, confirming institutional review of methodology and data handling. No human participants were recruited; therefore, informed consent pertained to secondary data use under the dataset's Apache 2.0 licence rather than participant interviews. The research avoids deceptive practices: the prototype clearly functions as an academic demonstrator, not a certified financial security product. Potential indirect ethical considerations include misuse of fraud-detection techniques for discriminatory profiling; the study does not incorporate demographic attributes, reducing but not eliminating algorithmic fairness concerns if deployed without oversight.

## 7.2 Legal and Regulatory Considerations

Use of the Cifer dataset complies with its Apache 2.0 licence, requiring attribution to Cifer AI and PaySim originators (Lopez-Rojas, Elmir and Axelsson, 2016). The project does not process personal data under UK GDPR definitions because records are synthetic and anonymised; nevertheless, a production deployment handling real transactions would require GDPR compliance, lawful basis for processing, data minimisation, retention limits, and potentially Financial Conduct Authority oversight depending on application context. Intellectual property in the original codebase and dissertation remains with the student author, while third-party libraries are used under their respective open-source licences.

## 7.3 Professional Responsibilities

As a computing professional-in-training, the researcher adhered to BCS codes of conduct emphasising accuracy, transparency, and public interest. Model limitations are documented rather than overstated. Stakeholders are informed that prototype outputs support research and education, not autonomous enforcement decisions. Secure deployment practices, including environment-based secret keys and health monitoring endpoints, were applied for the Render configuration. Supervisor feedback supported quality assurance.

## 7.4 Social Implications

Effective fraud detection benefits society by reducing financial crime losses and protecting consumers. However, false positives may inconvenience legitimate account holders, while false negatives permit criminal activity. Accessible open-source prototypes democratise fraud analytics for smaller fintech firms and researchers, potentially narrowing the resource gap identified in Chapter Two. Public trust depends on transparent communication about synthetic evaluation conditions and the prototype's non-production status.

---

# Chapter Eight: Conclusion and Recommendations

## 8.1 Summary of the Project

This research developed and evaluated a machine learning-based fraud detection prototype addressing the need for accessible, interpretable financial risk monitoring tools in academic and small-scale fintech contexts. Using the Cifer synthetic transaction dataset, the study implemented preprocessing, account legitimacy feature engineering, comparative training of four classifiers, and integration of the best-performing Decision Tree into a Flask web application with cloud deployment capability. The work responds directly to the research aim stated in Chapter One and stakeholder requirements identified at project inception.

## 8.2 Main Findings and Achievement of Objectives

The Decision Tree achieved 91.94% accuracy, 0.924 F1-score, and 0.987 ROC-AUC on balanced hold-out data, outperforming Logistic Regression (84.38%), XGBoost (82.24%), and Random Forest (79.22%). Account legitimacy indicators were essential; transaction-only models initially performed at chance level. All primary objectives—preprocessing, model comparison, algorithm selection, prototype development, and deployment preparation—were met. The research question is answered affirmatively: a lightweight ML prototype can effectively classify fraudulent transactions and present risk levels through a monitoring interface, subject to balanced-data evaluation constraints. The web artefact demonstrates that experimental results can be operationalised without enterprise infrastructure, fulfilling the practical dimension of the project brief.

## 8.3 Project Contribution

The project contributes a reproducible end-to-end pipeline, empirical comparison of four standard classifiers under transparent conditions, and a deployed research artefact linking experimental findings to practical inference. It demonstrates that domain-informed feature engineering can outweigh algorithmic complexity in constrained academic settings, offering a reference implementation for similar MSc projects.

## 8.4 Recommendations for Improvement and Future Work

Future work should evaluate models on naturally imbalanced data and multi-part temporal splits to better reflect production conditions. Cross-validation and fairness auditing across transaction types would strengthen reliability claims. Real-time streaming ingestion, explainability tools such as SHAP, and integration with rule-based systems could enhance analyst workflows. Scaling registry storage efficiently and retraining periodically would support live deployment. User acceptance testing with fraud analysts would validate interface usability beyond technical functional testing.

---

*Note: Replace figure placeholders with actual screenshots and the provided UI diagram before submission. Verify Harvard references in Chapters Six and Seven match your Chapter Two bibliography.*
