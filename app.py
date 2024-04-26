import streamlit as st
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

def processing_data(dataframe):
  needed_features = ['creditLimit', 'availableMoney', 'transactionDateTime',
                     'transactionAmount', 'merchantCountryCode', 'posEntryMode',
                     'posConditionCode','merchantCategoryCode', 'currentExpDate'
                     ,'accountOpenDate','dateOfLastAddressChange', 'cardCVV',
                     'enteredCVV', 'cardLast4Digits','transactionType',
                     'currentBalance', 'cardPresent']
  data = dataframe[needed_features]

  features_notobject = ["cardCVV", "enteredCVV", "cardLast4Digits"]
  for column in features_notobject:
    data[column] = pd.to_numeric(data[column], errors='coerce')

  features_date = ['transactionDateTime', 'currentExpDate', 'accountOpenDate',
                 'dateOfLastAddressChange']
  for column in features_date:
    data[column] = pd.to_datetime(data[column], errors='coerce')

  data['hour'] = data['transactionDateTime'].dt.hour
  data['day_of_week'] = data['transactionDateTime'].dt.day_name()

  data['accountAgeDays'] = (data['transactionDateTime'] - data['accountOpenDate']).dt.days
  data['accountAgeYears'] = data['accountAgeDays'] / 365.25

  data['cardLifeMonths'] = ((data['currentExpDate'].dt.year - data['accountOpenDate'].dt.year) * 12
                        + (data['currentExpDate'].dt.month - data['accountOpenDate'].dt.month))

  data['cardLifeYears'] = data['cardLifeMonths'] / 12
  data['daysSinceAddressChange'] = (data['transactionDateTime'] - data['dateOfLastAddressChange']).dt.days

  selected_columns = ['creditLimit', 'availableMoney','transactionAmount',
                    'merchantCountryCode','posEntryMode', 'posConditionCode',
                    'merchantCategoryCode', 'cardCVV', 'enteredCVV',
                    'cardLast4Digits', 'transactionType', 'currentBalance',
                    'cardPresent', 'hour','day_of_week', 'accountAgeYears',
                    'cardLifeYears', 'daysSinceAddressChange']
  return data[selected_columns]


def transforming_data(processed_dataframe):
  categorical_feature = ['merchantCountryCode','posEntryMode', 'posConditionCode',
                         'merchantCategoryCode','transactionType','day_of_week']

  encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  ct = ColumnTransformer([('one_hot_encoder', encoder, categorical_feature)],
                         remainder='drop')
  data_encoded = ct.fit_transform(processed_dataframe)
  data_encoded = pd.DataFrame(data_encoded, columns= ct.get_feature_names_out())

  other_feature = ['creditLimit', 'availableMoney','transactionAmount', 'cardCVV',
                  'enteredCVV', 'cardLast4Digits', 'currentBalance',
                 'cardPresent', 'hour', 'accountAgeYears',
                 'cardLifeYears', 'daysSinceAddressChange']

  final_data = pd.concat([processed_dataframe[other_feature], data_encoded], axis=1)

  return final_data


def main():
    st.title("FRAUD DETECTION IN TRANSACTIONS")
    st.subheader("Upload your transaction log (JSON format)")

    uploaded_file = st.file_uploader("Choose a file", type=["json", "txt"])
    if uploaded_file is not None:
        data = []
        for line in uploaded_file:
            line = line.decode('utf-8').strip()
            if line:
                data.append(json.loads(line))

        df = pd.DataFrame(data)
        account_numbers = df['accountNumber'].tolist()

        processed_data = processing_data(df)
        transformed_data = transforming_data(processed_data)
        reference_variable = ['creditLimit', 'availableMoney', 'transactionAmount', 'cardCVV',
                      'enteredCVV', 'cardLast4Digits', 'currentBalance', 'cardPresent',
                      'hour', 'accountAgeYears', 'cardLifeYears', 'daysSinceAddressChange',
                      'one_hot_encoder__merchantCountryCode_CAN',
                      'one_hot_encoder__merchantCountryCode_MEX','one_hot_encoder__merchantCountryCode_PR',
                      'one_hot_encoder__merchantCountryCode_US','one_hot_encoder__posEntryMode_02', 
                      'one_hot_encoder__posEntryMode_05','one_hot_encoder__posEntryMode_09', 'one_hot_encoder__posEntryMode_80',
                      'one_hot_encoder__posEntryMode_90','one_hot_encoder__posConditionCode_01',
                      'one_hot_encoder__posConditionCode_08', 'one_hot_encoder__posConditionCode_99',                   
                      'one_hot_encoder__merchantCategoryCode_airline','one_hot_encoder__merchantCategoryCode_auto',                    
                      'one_hot_encoder__merchantCategoryCode_cable/phone',
                      'one_hot_encoder__merchantCategoryCode_entertainment',
                      'one_hot_encoder__merchantCategoryCode_fastfood',
                      'one_hot_encoder__merchantCategoryCode_food',
                      'one_hot_encoder__merchantCategoryCode_food_delivery',
                      'one_hot_encoder__merchantCategoryCode_fuel',
                      'one_hot_encoder__merchantCategoryCode_furniture',
                      'one_hot_encoder__merchantCategoryCode_gym',
                      'one_hot_encoder__merchantCategoryCode_health',
                      'one_hot_encoder__merchantCategoryCode_hotels',
                      'one_hot_encoder__merchantCategoryCode_mobileapps',
                      'one_hot_encoder__merchantCategoryCode_online_gifts',
                      'one_hot_encoder__merchantCategoryCode_online_retail',
                      'one_hot_encoder__merchantCategoryCode_online_subscriptions',
                      'one_hot_encoder__merchantCategoryCode_personal care',
                      'one_hot_encoder__merchantCategoryCode_rideshare',
                      'one_hot_encoder__merchantCategoryCode_subscriptions',
                      'one_hot_encoder__transactionType_ADDRESS_VERIFICATION',
                      'one_hot_encoder__transactionType_PURCHASE',
                      'one_hot_encoder__transactionType_REVERSAL',
                      'one_hot_encoder__day_of_week_Friday',
                      'one_hot_encoder__day_of_week_Monday',
                      'one_hot_encoder__day_of_week_Saturday',
                      'one_hot_encoder__day_of_week_Sunday',
                      'one_hot_encoder__day_of_week_Thursday',
                      'one_hot_encoder__day_of_week_Tuesday',
                      'one_hot_encoder__day_of_week_Wednesday']
        
        reference_dataframe = pd.DataFrame({col: [0] * transforming_data(processed_data).shape[0]  for col 
                                            in reference_variable})
        
        
        for row in range(transformed_data.shape[0]):
            for feature in reference_variable:
                if feature in list(transformed_data.columns):
                    reference_dataframe.loc[row, feature]= transformed_data.loc[row, feature]
                else:
                    reference_dataframe.loc[row, feature] = 0

        reference_dataframe = reference_dataframe.astype(int)
        
        if st.button('Predict'):
           
            scaler = joblib.load('G:\My Drive\Fraud Detection\standard_scaler.joblib')
            input_data = scaler.transform(reference_dataframe)
            model = joblib.load(open(r"G:\My Drive\Fraud Detection\rf_model_new.joblib", 'rb'))
            predictions = model.predict(input_data)
           
            prediction_df = pd.DataFrame({'Account Number': account_numbers,'Prediction': 
                                          ['Fraudulent' if pred == 1 else 'Not Fraudulent' 
                                           for pred in predictions]})
            
            st.write('Prediction on the transactions')
            st.table(prediction_df)


if __name__ == '__main__':
    main()