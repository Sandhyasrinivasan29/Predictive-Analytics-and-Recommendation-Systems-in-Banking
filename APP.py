import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns


df_k = pd.read_csv("Loan_data_k.csv")

# Fix for _Scorer attribute error
import sklearn.metrics._scorer

class DummyScorer:
    def __init__(self, *args, **kwargs):
        pass

sklearn.metrics._scorer._Scorer = DummyScorer

# Load the model
with open("model_rf.pkl", 'rb') as f:
    random_forest = pickle.load(f)
model = random_forest.best_estimator_

# Load the scaler and label encoder
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("le.pkl", 'rb') as f:
    le = pickle.load(f)
#_____________________________________________
# Load the KMeans model
with open("kmeans.pkl", 'rb') as f:
    Kmeans_model = pickle.load(f)

with open("scaler_kmeans.pkl", 'rb') as f:
    scaler_k = pickle.load(f)

with open("le_kmeans.pkl", 'rb') as f:
    le_k = pickle.load(f)
#______________________________________________

# Define feature columns
Features = [
    'Age', 'Income', 'Credit_score', 'Credit_score_category',
    'Loan_amount', 'Interest_rate', 'Loan_term', 'Loan_type',
    'Box_cox_Debit_income'
]

# Categorical mapping
categorical_mapping = {
    "Credit_score_category": {"Excellent": 0, "Fair": 1, "Good": 2, "Poor": 3},
    "Loan_type": {"Auto": 0, "Business": 1, "Education": 2, "Mortgage": 3, "Personal": 4},
}

# Streamlit UI elements
st.markdown("<h1 style='display: flex; align-items: center; font-size: 20px; margin: 0;'>PREDICTIVE ANALYTICS AND RECOMMENDATION SYSTEM IN BANKING</h1>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title="Banking Services",  # Updated menu title to reflect a bank app
        options=["LOAN DEFAULT PREDICTION", "CUSTOMER SEGMENTATION"],
        icons=["cash-stack", "people-fill", "bag-check-fill"],  # Updated icons for a banking theme
        menu_icon="bank",  # Icon representing the menu itself
        default_index=0,  # Set default selected option
        styles={
            "container": {"padding": "5px", "background-color": "#F8F9FA"},  # Light gray container
            "icon": {"color": "#ff0000", "font-size": "20px"},  # Blue icons
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "5px",
                "color": "#495057",
                "border-radius": "5px",
                "background-color": "#E9ECEF"
            },  # Neutral button style
            "nav-link-hover": {"background-color": "#007BFF", "color": "white"},  # Hover effect
            "nav-link-selected": {
                "background-color": "#007BFF",
                "color": "white",
                "font-weight": "bold",
                "border": "1px solid #0056b3"
            }  # Selected option style
        }
    )

#-------------------------------------LOAN PREDICTION--------------------------------------------------------------
if selected == "LOAN DEFAULT PREDICTION":
    st.title("Loan Default Prediction")
    # Collect user input
    age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
    income = st.number_input("Enter Income", min_value=1000, max_value=1000000, value=50000)
    credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=650)
    credit_score_category = st.selectbox("Select Credit Score Category", options=["Excellent", "Fair", "Good", "Poor"])
    loan_amount = st.number_input("Enter Loan Amount", min_value=1000, max_value=1000000, value=100000)
    interest_rate = st.number_input("Enter Interest Rate (%)", min_value=1.5, max_value=12.5, value=5.0)
    loan_term = st.number_input("Enter Loan Term (Years)", min_value=1, max_value=5, value=3)
    loan_type = st.selectbox("Select Loan Type", options=["Auto", "Business", "Education", "Mortgage", "Personal"])
    debit_amt = st.number_input("Enter Debt to Income Ratio", min_value=0.0, max_value=50.0, value=1.0)

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Credit_score': [credit_score],
        'Credit_score_category': [credit_score_category],
        'Loan_amount': [loan_amount],
        'Interest_rate': [interest_rate],
        'Loan_term': [loan_term],
        'Loan_type': [loan_type],
        'Box_cox_Debit_income': [np.log1p(debit_amt)],  # Log transformation applied directly
    })

    # Apply categorical mapping
    input_data['Credit_score_category'] = input_data['Credit_score_category'].map(categorical_mapping["Credit_score_category"])
    input_data['Loan_type'] = input_data['Loan_type'].map(categorical_mapping["Loan_type"])

    # Check if all required columns are present
    if all(col in input_data.columns for col in Features):
        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Prediction logic
        if st.button("Predict Loan Default Risk"):
            prediction = model.predict(scaled_data)
            if prediction[0] == 1:
                st.markdown(
                    f"<h3 style='color: #FF0000;'>⚠️ High probability of Default</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='color: #00FF00;'>✅ Low probability of Default</h3>",
                    unsafe_allow_html=True
                )
    else:
        st.error("Some required input features are missing.")
#-------------------------------------CUSTOMER SEGMENTATION--------------------------------------------------------------

# Customer segmentation
if selected == "CUSTOMER SEGMENTATION":

        # st.subheader("CUSTOMER SEGMENTATION")
        try:
            transaction_amount = float(st.text_input("Transaction Amount", "1.0"))
            transaction_frequency = float(st.text_input("Transaction Frequency", "1.0"))
        except ValueError:
            st.error("Please enter valid numbers for transaction amount and frequency.")
            transaction_amount = 1.0
            transaction_frequency = 0.0
        
        transaction_type = st.selectbox("Transaction_Type", ["Deposit", "Withdrawal"])

        # Refit LabelEncoder with all possible labels
        le.fit(["Deposit", "Withdrawal"])
        transaction_type_encoded = le.transform([transaction_type])[0]

        if st.button("Predict Cluster"):
            input_data = {
                "Transaction_amt": [transaction_amount],  # Corrected feature name
                "Transaction_freq": [transaction_frequency],  # Corrected feature name
                "Transaction_Type_Encoded": [transaction_type_encoded]
            }

            input_df = pd.DataFrame(input_data)

            input_scaled_data = scaler_k.transform(input_df)

            predicted_customer = Kmeans_model.predict(input_scaled_data)

            df_k["Transaction_Type_Encoded"] = le.transform(df_k["Transaction_type"])
            df_scaled = scaler_k.transform(df_k[["Transaction_amt", "Transaction_freq", "Transaction_Type_Encoded"]])
            df_k["Clusters"] = Kmeans_model.fit_predict(df_scaled)

            # Show predicted customer cluster
            st.markdown(
                f"<h3 style='color: #ff0000;'>The customer belongs to cluster: {predicted_customer[0]}</h3>",
                unsafe_allow_html=True
            ) 
            if predicted_customer==0 or predicted_customer==6 :
                st.write("""Interpretation:
                            High-Value, Low-Frequency Customers: Customers who make large but infrequent transactions. These customers are likely to make significant financial moves occasionally, such as large investments, high-value purchases, or substantial transfers.
                            Potential High Net-Worth Individuals (HNWIs): Given the high transaction amounts, this cluster might include high net-worth individuals or entities that engage in large financial activities but do not do so frequently.""")
            elif predicted_customer==2 or predicted_customer==3:
                st.write("""Interpretation:
                            Balanced Transaction Behavior: Customers in Cluster 2 and 3 exhibit balanced transaction behavior with neither high nor low transaction values and frequencies. They are likely to be steady users of financial services who engage in routine banking activities.
                            Potential Mid-Tier Customers: These customers may fall into the mid-tier category, not engaging in high-value or highly frequent transactions but maintaining consistent financial activity.""")
            elif predicted_customer==1 or predicted_customer==7:
                st.write("Interpretation:")
                         
                st.write("Lower Average Transaction Amount: Cluster has a lower average transaction amount , indicating that the transactions in this cluster are relatively small.")
                st.write("High Transaction Frequency: With an average transaction frequency , customers in this Cluster  make transactions quite frequently. This suggests that these customers are very active and frequently engage with their financial services, making regular, smaller transactions.")
            
            
            elif predicted_customer==4 or predicted_customer==5:
                st.write("""Interpretation:
                         High-Value, High-Frequency Customers: Cluster 4 and 5 represents customers who not only make large transactions but do so frequently. These customers are likely very important to the financial institution as they contribute significantly both in terms of transaction volume and frequency.
                         Potential VIPs or Business Accounts: Given the high transaction amounts and frequencies, this cluster might include VIP customers or business accounts. These customers may require personalized services, dedicated account managers, and tailored financial products to meet their high-value needs.""")
            

            
            
            
            
            
            # Summarize clusters
            cluster_summary = df_k.groupby("Clusters").agg(
                Average_Transaction_Amount=("Transaction_amt", "mean"),
                Average_Transaction_Frequency=("Transaction_freq", "mean"),
            ).reset_index()

            st.markdown(
                f"<h3 style='color: #D3D3D3;'>CLUSTER SUMMARY</h3>",
                unsafe_allow_html=True
            )

            # Display the summary table
            st.write(cluster_summary)

            # Create a dual-axis plot for cluster summary
            fig, ax1 = plt.subplots(figsize=(12, 8))

            ax2 = ax1.twinx()
            ax1.bar(cluster_summary['Clusters'], cluster_summary['Average_Transaction_Amount'], color='g', alpha=0.6, width=0.4, align='center')
            ax2.plot(cluster_summary['Clusters'], cluster_summary['Average_Transaction_Frequency'], color='b', marker='o')

            ax1.set_xlabel('Clusters')
            ax1.set_ylabel('Average Transaction Amount', color='g')
            ax2.set_ylabel('Average Transaction Frequency', color='b')
            ax1.set_title('Cluster Summary')

            ax1.legend(['Avg Transaction Amount'], loc='upper left')
            ax2.legend(['Avg Transaction Frequency'], loc='upper right')

            st.pyplot(fig)
#_____________________________________________________________________________________________
