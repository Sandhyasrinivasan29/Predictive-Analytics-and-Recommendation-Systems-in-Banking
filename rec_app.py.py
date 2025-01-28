import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
import pickle

# Load pre-trained KNN model
with open("models.pkl", "rb") as f:
    model = pickle.load(f)


# Load dataset
df = pd.read_csv('Loan_data_k.csv')

# Mapping product IDs to product names
product_mapping = {
    "P001": "Savings Account", "P002": "Premium Checking Account", "P003": "Credit Card", 
    "P004": "Personal Loan", "P005": "Home Loan", "P006": "Auto Loan", "P007": "Business Loan", 
    "P008": "Student Loan", "P009": "Investment Fund", "P010": "Retirement Plan", "P011": "Insurance Policy", 
    "P012": "Mutual Fund", "P013": "Bond", "P014": "Certificate of Deposit", "P015": "Home Equity Line", 
    "P016": "Mortgage Refinance", "P017": "Business Credit Line", "P018": "Auto Refinance", 
    "P019": "Home Improvement Loan", "P020": "Gold Loan", "P021": "Cash Credit", "P022": "Short-Term Loan", 
    "P023": "Long-Term Loan", "P024": "Travel Loan", "P025": "Medical Loan", "P026": "Emergency Loan", 
    "P027": "Holiday Loan", "P028": "Debt Consolidation Loan", "P029": "Small Business Loan", 
    "P030": "Agricultural Loan", "P031": "Technology Loan", "P032": "Education Savings Plan", 
    "P033": "Wealth Management", "P034": "Stock Investment", "P035": "Real Estate Investment", 
    "P036": "International Investment", "P037": "Fixed Deposit", "P038": "Recurring Deposit", 
    "P039": "Loan Against Property", "P040": "Gold Investment", "P041": "Retirement Savings", 
    "P042": "High-Yield Savings Account", "P043": "Money Market Account", "P044": "Insurance Savings", 
    "P045": "Pension Plan", "P046": "Child Education Fund", "P047": "Healthcare Savings", 
    "P048": "Property Investment", "P049": "Auto Insurance"
}

df["Product_name"] = df["Product_id"].map(product_mapping)

# Mapping interaction types
interaction_type_mapping = {'Viewed': 1, 'Clicked': 2, 'Purchased': 3}
df["Interaction_Type"] = df["Interaction_type"].map(interaction_type_mapping)

# Surprise Reader for Surprise model compatibility
reader = Reader(rating_scale=(1, 3))

# Recommendation function
def recommend_products(customer_id, model, interaction_data, product_mapping, n=5):
    all_products = set(interaction_data['Product_id'].unique())

    # Get customer's interacted products
    interacted_products = set(interaction_data[interaction_data['Customer_id'] == customer_id]['Product_id'])
    print(f"Customer {customer_id} has interacted with: {interacted_products}")  # Debugging line

    # If the customer has no interactions, suggest popular products from all available products
    products_to_predict = list(all_products - interacted_products) if customer_id in interaction_data['Customer_id'].values else list(all_products)

    print(f"Products to predict for customer {customer_id}: {products_to_predict}")  # Debugging line

    # If no products to predict, return empty
    if not products_to_predict:
        return pd.DataFrame(columns=['Product_Id', 'Product_Name'])

    # Predictions for each product
    predictions = [model.predict(customer_id, product_id) for product_id in products_to_predict]
    
    print(f"Predictions for customer {customer_id}: {predictions}")  # Debugging line
    
    # Sort predictions by estimated rating (highest first)
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get top recommended product IDs and names
    recommended_product_ids = [pred.iid for pred in top_n_predictions]
    recommended_products = pd.DataFrame({
        'Product_Id': recommended_product_ids,
        'Product_Name': [product_mapping.get(pid, 'Unknown') for pid in recommended_product_ids]
    })
    return recommended_products

# Sidebar Navigation
options = ["PRODUCT RECOMMENDATIONS"]
selected = st.sidebar.selectbox("Select a page", options)

# Check if "PRODUCT RECOMMENDATIONS" page is selected
if selected == "PRODUCT RECOMMENDATIONS":
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("PRODUCT RECOMMENDATIONS")

        # Input field for Customer ID
        customer_id = st.text_input("Enter Customer ID", "")

        # Button to get recommendations
        if st.button("Get Recommendations"):
            if customer_id:
                if customer_id in df['Customer_id'].values:
                    recommendations = recommend_products(customer_id, model, df, product_mapping)

                    if not recommendations.empty:
                        st.markdown(
                            f"""
                            <div style="background-color: #d9f9d9; padding: 10px; border-radius: 10px;">
                                <h3 style="color: #000000; text-align: center;">
                                    Top Recommended Products for {customer_id}:
                                </h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.dataframe(recommendations)
                    else:
                        st.write("No recommendations available.")
                else:
                    st.error("Invalid Customer ID. Please enter a valid Customer ID.")
            else:
                st.warning("Please enter a valid Customer ID.")
