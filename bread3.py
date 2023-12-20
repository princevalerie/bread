import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load Data
df = pd.read_csv("bread basket.csv")

# Data Preparation
unique_items = df['Item'].unique()
period_day_options = df['period_day'].unique()
weekday_weekend_options = df['weekday_weekend'].unique()

# Streamlit App
st.title('Product Association Analysis')

# Sidebar for filters
st.sidebar.header('Filters')

# User Input
selected_item = st.sidebar.selectbox('Select an item:', unique_items)
selected_period_day = st.sidebar.selectbox('Select a period day:', period_day_options)
selected_weekday_weekend = st.sidebar.selectbox('Select a weekday or weekend:', weekday_weekend_options)

# Filter data based on user input
filtered_data = df

# Display filtered data
st.sidebar.subheader(f'Data for {selected_item}, {selected_period_day}, and {selected_weekday_weekend}:')
st.sidebar.write(filtered_data)

# Data Preparation
filtered_data['Item'] = filtered_data['Item'].str.strip()
filtered_data['Item'] = filtered_data['Item'].str.lower()

transactions = filtered_data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Number of Items')

# Table Transformation
table = transactions.pivot_table(index='Transaction', columns='Item', values='Number of Items', aggfunc='sum').fillna(0)

def hot_encode(x):
    if(x == 0):
        return False
    if(x > 0):
        return True

final_table = table.applymap(hot_encode)

# Import Machine Learning
frequence = apriori(final_table, min_support=0.015, use_colnames=True)
product_association = association_rules(frequence, metric='confidence', min_threshold=0.5).sort_values('confidence', ascending=False).reset_index(drop=True)

# Convert frozenset to list
product_association['antecedents'] = product_association['antecedents'].apply(lambda x: list(x))
product_association['consequents'] = product_association['consequents'].apply(lambda x: list(x))

# Display the association rules
st.subheader('Association Rules with Confidence >= 0.5')
st.dataframe(product_association)

# Button for searching consequents
if st.button('Search Consequents'):
    # Display the consequents for the selected rule
    if not product_association.empty:
        selected_rule_consequents = product_association['consequents'].iloc[0]
        st.subheader(f'Consequents for the Top Rule:')
        st.write(f"Maka orang juga membeli",selected_rule_consequents[0])
    else:
        st.subheader('No rules found with confidence >= 0.5.')
