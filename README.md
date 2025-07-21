## 1. Project Background

Following a series of discussions with key stakeholders, we have aligned on the business objective and requirements for a customer classification project aimed at supporting a targeted retention campaign for a Telco company.

The company is planning to launch a voucher distribution campaign, offering a **20% discount** to a select group of customers. The primary goal of this campaign is to **maintain the monthly customer retention rate above 85%**.

## 2. Business Requirements

- **Voucher Allocation Constraints:**
  - A maximum of **500 vouchers** can be distributed during this campaign.
  - Each voucher provides a **20% service fee discount**.

- **Targeting Strategy:**
  - **Primary Objective:** Prioritize sending vouchers to customers with the **highest likelihood of churn**.
  - **Secondary Objective:** Accept that some vouchers may be sent to customers who are likely to stay with the service even without incentives. These cases are considered part of a broader **customer care initiative**.

## 3. Analytical Approach

To support this business goal, we will develop a **customer churn prediction model**. The output of this model — a list of high-risk customers — will be used by the Marketing team to guide voucher distribution.

## 4. Model Objective and Evaluation Metrics

Given the nature of the campaign and business constraints, the model will be optimized based on the following principles:

- **High Recall (Sensitivity):**  
  The model must identify as many potential churners as possible to maximize the effectiveness of the campaign. Missing true churners would mean lost opportunities for retention.

- **Balanced Precision:**  
  While high recall is essential, it is also important to **maintain a reasonable level of precision**. Over-predicting churn (i.e., low precision) would lead to a high number of vouchers being allocated to customers who would have stayed regardless — thus reducing cost-efficiency.

Hence, model evaluation will primarily focus on:
- **Recall** – as the key metric
- **Precision** – to control budget waste
- **F1-Score or PR-AUC** – as secondary metrics to ensure a balanced performance

## 5. Model Development
View [model_experiment.ipynb]['notebooks/model_experiment.ipynb'] for more detail 
### EDA Highlights

Dataset: **7,043 records**, **21 features** (excluding `customerID`). Grouped into:
- **Demographics:** `gender`, `seniorcitizen`, `partner`, `dependents`
- **Service Features:** e.g., `internetservice`, `phoneservice`, `streamingtv`
- **Billing:** `tenure`, `monthlycharges`, `totalcharges`

Insights:
- Standardized text to lowercase; fixed `totalcharges` data type.
- Created new feature combining phone/internet services.
- After cross-checking features like `phoneservice` and `internetservice`, `multipleline` we identified three customer groups:
  - **Phone service only**
  - **Internet service only**
  - **Multiple services**
  We created a new feature to label each observation accordingly.

- **Churn rate:** 26.5% (moderately imbalanced dataset).
- High churn associated with:
  - **Month-to-month contracts**
  - **Electronic check payments**
  - **Paperless billing**
  - **Senior citizens** and those without partners or dependents
- Identified multicollinearity between `tenure`, `monthlycharges`, and `totalcharges`; dropped `totalcharges` from modeling.

### Modeling Summary

We experimented with:
- **Logistic Regression (L2 Regularization)**
- **Decision Tree**
- **XGBoost**

Each algorithm went through:
1. Initial training
2. Evaluation
3. Hyperparameter tuning
4. Final model selection

**Results:**
- **Logistic Regression** achieved the best **recall (train: 0.79, test: 0.83)**, with acceptable **precision(train:0.51, test: 0.53)** and **F1-score**. Saved `BestLogModel` as `logistic_model.pkl`.
- **Decision Tree** showed high variance and overfitting.
- **XGBoost** offered flexibility but failed to outperform Logistic Regression on recall.

## 6. Model Deployment

Final model deployed via **Gradio** (`app.py`).

Users upload a pre-processed CSV. The app returns churn predictions along with customer IDs for marketing action.
- **Demo Video**
[![Watch the demo video](/media/thumbnail.png)](/media/demo_video.mp4)
- **Pipeline of Web App**
![customer_churn_prediction_webapp_pipeline](/media/customer_churn_prediction_webapp.png)