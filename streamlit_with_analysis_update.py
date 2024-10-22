import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import warnings
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os

warnings.filterwarnings("ignore")


directory = 'D:\Foundations_of_DataScience\Projects\Mid_project\Dataset10\CMSE830_midproject' 
dfs = []
for i in range(9):
    file_path = os.path.join(directory, f'chunk_{i}.csv')
    df = pd.read_csv(file_path)
    dfs.append(df)

data_init = pd.concat(dfs, ignore_index=True)
print(data_init.columns)
# print(data_init.columns)


# # Load data (you can load the same data used in the notebook)
# @st.cache_data
# def load_data():
#     data_init = pd.read_csv(r'D:\Foundations_of_DataScience\Projects\Mid_project\Dataset10\CMSE830_midproject\loan_data_2015.csv')  # Replace with your file path if necessary
#     return data_init

# data_init = load_data()
# data=data_init.copy()
data=data_init[['id','loan_amnt', 'funded_amnt', 'revol_bal','total_rev_hi_lim',
           'int_rate','installment','total_pymnt','total_rec_late_fee','recoveries','last_pymnt_amnt','out_prncp','total_rec_prncp','total_rec_int',
           'delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','total_acc','pub_rec',
           'dti','annual_inc','verification_status',
           'revol_util',
           'emp_length','home_ownership',
           'pymnt_plan','grade','sub_grade','loan_status','purpose','acc_now_delinq','mths_since_last_delinq','addr_state']].copy()

mask = (data['loan_status'] != 'Current') & (data['loan_status'] != 'Issued') & (data['loan_status'] != 'Late (16-30 days)') & (data['loan_status'] != 'In Grace Period')
data = data[mask]
data['risk'] = np.where((data['loan_status'] == 'Charged Off') | (data['loan_status'] == 'Late (31-120 days)') | (data['loan_status'] == 'Default'), 1, 0)
data.drop('loan_status', axis=1, inplace=True)

loan_amounts_balances = pd.DataFrame({
    'Column Name': ['id', 'loan_amnt', 'funded_amnt', 'revol_bal', 'total_rev_hi_lim', 'out_prncp', 'total_rec_prncp'],
    'Description': [
        'Unique identifier for each loan.', 
        'The total amount of money that the borrower requested.', 
        'The actual amount of money funded by investors.', 
        'Total credit revolving balance (amount owed on revolving accounts).',
        'The total revolving high credit/credit limit.', 
        'Remaining outstanding principal balance for total amount funded.',
        'Principal amount received to date.'
    ]
})

payment_behavior = pd.DataFrame({
    'Column Name': ['installment', 'total_pymnt', 'total_rec_late_fee', 'recoveries', 'last_pymnt_amnt'],
    'Description': [
        'Monthly payment amount owed by the borrower.',
        'Total payments (including principal, interest, late fees, etc.) received to date.',
        'Total late fees received.',
        'Post-charge off gross recovery amount.',
        'Amount of the last payment made by the borrower.'
    ]
})

credit_history = pd.DataFrame({
    'Column Name': ['delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'total_acc', 'pub_rec', 'acc_now_delinq', 'mths_since_last_delinq'],
    'Description': [
        'The number of 30+ days past-due incidences of delinquency in the borrower’s credit file in the last 2 years.',
        'The date the borrower’s earliest reported credit line was opened.',
        'Number of credit inquiries in the last 6 months.',
        'The number of open credit lines in the borrower’s credit file.',
        'The total number of credit lines currently in the borrower’s credit file.',
        'Number of derogatory public records (e.g., bankruptcies, tax liens).',
        'The number of accounts currently delinquent.',
        'Months since the borrower’s last delinquency.'
    ]
})

dti_income = pd.DataFrame({
    'Column Name': ['dti', 'dti_joint', 'annual_inc', 'verification_status'],
    'Description': [
        'Debt-to-income ratio of the borrower.',
        'Debt-to-income ratio for joint applications.',
        'The self-reported annual income provided by the borrower during loan application.',
        'Indicates whether the income was verified.'
    ]
})

credit_utilization = pd.DataFrame({
    'Column Name': ['revol_util', 'mths_since_rcnt_il'],
    'Description': [
        'Revolving line utilization rate, or the amount of credit used relative to the credit limit.',
        'Months since most recent installment loan.'
    ]
})

personal_info = pd.DataFrame({
    'Column Name': ['emp_length', 'home_ownership', 'addr_state', 'application_type', 'emp_title', 'title'],
    'Description': [
        'Number of years the borrower has been employed.',
        'The borrower’s home ownership status (e.g., Rent, Own, Mortgage).',
        'The state provided by the borrower in the loan application.',
        'Indicates whether the loan application is individual or joint.',
        'Job title provided by the borrower.',
        'Loan title provided by the borrower.'
    ]
})

# Main Title
st.title("Credit risk analysis Dashboard")
my_analysis = st.sidebar.radio("Select Section", ["My Analysis", "Initial Data Analysis (IDA)", "Exploratory Data Analysis (EDA)"])

# My Analysis Section
if my_analysis == "My Analysis":
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Initial Data Analysis", "Exploratory Data Analysis", "Conclusions and Future Scope"])

    # Problem Statement
    with tab1:
        st.header("Problem Statement")
        st.write("""
    A bank is facing an increasing number of defaults on loans, and needs to improve its credit risk modeling in order to 
    better predict which customers are likely to default on their loans in the future.

    The bank needs to identify customers who are at high risk of defaulting on their loans in order to take appropriate action, 
    such as increasing the interest rate on the loan or declining the loan application. This will help the bank reduce its overall risk
    and increase its profitability.

    By accurately identifying high-risk customers, the bank can also improve its customer relationships by being more transparent 
    about the terms of the loan. Additionally, the bank can use the results of the credit risk modeling to inform its 
    marketing and sales strategies, targeting higher-risk customers with more appropriate products and services.

    The goal is to accurately identify high-risk loans in order to take appropriate action, 
    such as increasing the interest rate on the loan or declining the loan application, 
    which will help the bank reduce its overall risk and increase its profitability.
    
    """)
        st.subheader("Columns Description")
    
        # Loan Amounts and Balances
        st.markdown("### Loan Amounts and Balances")
        st.table(loan_amounts_balances)

        # Payment Behavior
        st.markdown("### Payment Behavior")
        st.table(payment_behavior)

        # Credit History
        st.markdown("### Credit History")
        st.table(credit_history)

        # Debt-to-Income (DTI) and Income
        st.markdown("### Debt-to-Income (DTI) and Income")
        st.table(dti_income)

        # Credit Utilization
        st.markdown("### Credit Utilization")
        st.table(credit_utilization)

        # Personal Information
        st.markdown("### Personal Information")
        st.table(personal_info)


    # Initial Data Analysis (IDA)
    with tab2:
        st.header("Initial Data Analysis (IDA)")
        st.write("This section provides a preliminary overview of the dataset's characteristics through descriptive statistics.")
        
        
        # Displays summary statistics
        st.write("### Summary Statistics")
        st.write("Displays the distribution and spread of numerical features in the dataset.")
        st.write(data.describe().T)
        st.write("""**Inference**: The summary statistics provide an overview of central tendencies (mean, median) and 
                 variability (std dev), helping to spot potential outliers or skewed distributions.""")
        
        # Summary of missing values
        missing_cols = data.isnull().sum()[data.isnull().sum() > 0]
        st.write("### Missing Values (Only Columns with Missing Data)")
        st.write(missing_cols)
        
        # Visualizations for missing data
        missing_cols = data.columns[data.isnull().sum() > 0]
        if not missing_cols.empty:
            st.write("### Missing Data Visualization")
            st.write("Shows which features have missing data and their density.")
            fig, ax = plt.subplots()
            sns.heatmap(data[missing_cols].isnull(), cbar=False, ax=ax)
            plt.title('Missing Data Heatmap (Only Columns with Missing Data)')
            st.pyplot(fig)
        else:
            st.write("No missing data found.")
        # st.write("""**Inference**: Features with large amounts of missing data might require more careful imputation, removal, 
        #          or flagging for risk in further analyses.""")


        # categorical_columns = data.select_dtypes(include=['object']).columns
        # encoder = OneHotEncoder(sparse=False, drop='first')
        # encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

        # Combine numeric and encoded categorical columns
        # data_combined = pd.concat([data_numeric_imputed, encoded_data], axis=1)

        st.write("### Histograms and Interactive plot")
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        def plot_histogram_with_kde(column_name):
            data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
            column_data = data[column_name].dropna()
            histogram = go.Histogram(
                x=column_data,
                nbinsx=30,
                name=f'{column_name} Distribution',
                opacity=0.75,
                marker=dict(color='blue'),
            )
            kde = gaussian_kde(column_data)
            x_kde = np.linspace(column_data.min(), column_data.max(), 100)
            y_kde = kde(x_kde)
            smoothing_curve = go.Scatter(
                x=x_kde,
                y=y_kde * (len(column_data) * (x_kde[1] - x_kde[0])),  # Adjust KDE to match histogram
                mode='lines',
                name=f'Smoothing Curve (KDE) for {column_name}',
                line=dict(color='orange', width=2)
            )
            fig = go.Figure(data=[histogram, smoothing_curve])
            fig.update_layout(
                title=f'Distribution of {column_name} with Smoothing Curve',
                xaxis_title=column_name,
                yaxis_title='Count',
                template='plotly_white'
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig)

        # Streamlit layout
        st.title("Numerical Feature Distribution with KDE")
        for column in data.columns:
            try:
                data[column] = pd.to_datetime(data[column], errors='coerce')
            except:
                continue
        # Loop through each numerical column and plot
        for column in numerical_columns:
            if column!='id':
                st.subheader(f"Column: {column}")
                plot_histogram_with_kde(column)

        

    # Exploratory Data Analysis (EDA)
    with tab3:
        st.header("Exploratory Data Analysis (EDA)")
        st.write("This section delves deeper into the relationships among variables.")

        # Bivariate Analysis
        st.write("### Bivariate Analysis")
        
        # Dropping the 'id' column if present
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
        
        # Calculating the correlation matrix
        correlation = data.corr()

        # Filtering correlation values where the absolute value is greater than 0.5
        filtered_corr = correlation[(correlation.abs() > 0.5) & (correlation != 1.0)].dropna(how='all', axis=0).dropna(how='all', axis=1)

        # Plotting the heatmap
        if not filtered_corr.empty:
            sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Filtered Correlation Heatmap (|correlation| > 0.5)')
            st.pyplot()
        else:
            st.write("No correlations found with an absolute value greater than 0.5.")
        
        # Scatter plots for selected pairs
        st.write("### Scatter Plots")
        scatter_pairs = [('loan_amnt', 'annual_inc'), ('dti', 'int_rate')]  # Predefined pairs
        selected_pair = st.selectbox("Select the scatter plot pair:", scatter_pairs)

        x_axis, y_axis = selected_pair

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_axis, y=y_axis, hue='risk', alpha=0.6)
        plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend(title='Risk')
        st.pyplot()
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Regression Analysis
        st.write("### Regression Analysis")
        pair_columns = ['loan_amnt', 'annual_inc', 'dti', 'int_rate']
        g = sns.pairplot(data[pair_columns])
        
        regression_results = {}
        for i, x_var in enumerate(pair_columns):
            for j, y_var in enumerate(pair_columns):
                if i < j:
                    x_data = data[x_var].dropna()
                    y_data = data[y_var].dropna()
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    ax = g.axes[i, j]
                    x_range = np.linspace(x_data.min(), x_data.max(), 100)
                    y_range = slope * x_range + intercept
                    ax.plot(x_range, y_range, color='red', linewidth=2)
                    
                    regression_results[f'{x_var} vs {y_var}'] = {'slope': slope, 'intercept': intercept}

        plt.show()
        st.pyplot()

        # Display regression results
        for pair, result in regression_results.items():
            st.write(f"{pair}: Slope = {result['slope']:.2f}, Intercept = {result['intercept']:.2f}")

    # Conclusions and Future Scope
    with tab4:
        st.header("Conclusions and Future Scope")
        st.write("""Through comprehensive analyses of the dataset, we have explored various factors influencing loan risk, 
                 ultimately aiming to develop a predictive model for assessing borrower reliability.""")
        st.write("""The insights gleaned from our analyses form a solid foundation for developing a predictive model to assess loan risk. 
                 Key predictors identified include annual income, loan amount, DTI, interest rates, and borrower credit history. 
                 By quantifying these relationships, we can construct a model that not only predicts the likelihood of loan default 
                 but also assists lenders in making informed decisions about whom to extend credit to.""")
        st.write("""To enhance model accuracy, we will consider feature engineering, incorporating interactions between variables, 
                 and leveraging machine learning algorithms suitable for classification tasks. By continually refining our approach 
                 based on the insights gained, we aim to create a robust predictive tool that mitigates risk and improves lending outcomes.""")

# Column Descriptions Tab
if my_analysis == "Initial Data Analysis (IDA)":
    st.sidebar.title('Column Descriptions')
    st.sidebar.write('Select a column to view its details:')
    selected_column = st.sidebar.selectbox('Select Column', data.columns)

    if selected_column:
        column_data = data[selected_column]
        
        st.subheader(f'Description for {selected_column}')
        st.write(f"**Data type**: {column_data.dtype}")
        st.write(f"**Mean**: {column_data.mean() if pd.api.types.is_numeric_dtype(column_data) else 'N/A'}")
        st.write(f"**Min**: {column_data.min()}")
        st.write(f"**Median**: {column_data.median() if pd.api.types.is_numeric_dtype(column_data) else 'N/A'}")
        st.write(f"**Max**: {column_data.max()}")
        st.write(f"**Count**: {column_data.count()}")
        st.write(f"**Missing values**: {column_data.isnull().sum()}")
        
        # Outliers detection using IQR
        if pd.api.types.is_numeric_dtype(column_data):
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = column_data[(column_data < (Q1 - 1.5 * IQR)) | (column_data > (Q3 + 1.5 * IQR))]
            st.write(f"**Outliers count**: {len(outliers)}")
            
        # Missing value imputation
        if column_data.isnull().sum() == 0:
            st.write(f"**Imputation method**: 'None applied'")
        elif pd.api.types.is_numeric_dtype(column_data):
            st.write(f"**Imputation method**: Median")
        else:
            st.write(f"**Imputation method**: Mode")

# EDA Tab
if my_analysis == "Exploratory Data Analysis (EDA)":
    st.sidebar.title('Exploratory Data Analysis (EDA)')
    eda_tab = st.sidebar.selectbox('Choose analysis', ['Univariate Analysis', 'Bivariate Analysis'])

    # Univariate Analysis
    if eda_tab == 'Univariate Analysis':
        st.header('Univariate Analysis')
        
        selected_column = st.sidebar.selectbox('Select Column for Univariate Analysis', data.columns)
        plot_type = st.sidebar.selectbox('Select Plot Type', ['Histogram', 'Box Plot', 'KDE'])
        color_palette = st.sidebar.selectbox('Select Color Palette', sns.color_palette().as_hex())

        if selected_column:
            fig, ax = plt.subplots()
            if plot_type == 'Histogram':
                sns.histplot(data[selected_column], kde=False, color=color_palette, ax=ax)
            elif plot_type == 'Box Plot':
                sns.boxplot(data[selected_column], color=color_palette, ax=ax)
            elif plot_type == 'KDE':
                sns.kdeplot(data[selected_column], color=color_palette, ax=ax)
            st.pyplot(fig)

    # Bivariate Analysis
    elif eda_tab == 'Bivariate Analysis':
        st.header('Bivariate Analysis')

        x_axis = st.sidebar.selectbox('Select X Axis', data.columns)
        y_axis = st.sidebar.selectbox('Select Y Axis', data.columns)
        plot_type_bi = st.sidebar.selectbox('Select Plot Type', ['Scatterplot', 'Line Plot', 'Hexbin', 'Joint Plot'])
        
        if x_axis and y_axis:
            fig, ax = plt.subplots()
            
            if plot_type_bi == 'Scatterplot':
                sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            elif plot_type_bi == 'Line Plot':
                sns.lineplot(x=data[x_axis], y=data[y_axis], ax=ax)
            elif plot_type_bi == 'Hexbin' and pd.api.types.is_numeric_dtype(data[x_axis]) and pd.api.types.is_numeric_dtype(data[y_axis]):
                ax.hexbin(data[x_axis], data[y_axis], gridsize=30, cmap='Blues')
                plt.colorbar(ax.collections[0], ax=ax)
            elif plot_type_bi == 'Joint Plot':
                sns.jointplot(x=data[x_axis], y=data[y_axis], kind="scatter")
            
            # Regression Line (if both axes are numeric)
            if plot_type_bi in ['Scatterplot', 'Line Plot']:
                if pd.api.types.is_numeric_dtype(data[x_axis]) and pd.api.types.is_numeric_dtype(data[y_axis]):
                    X = data[x_axis].values.reshape(-1, 1)
                    Y = data[y_axis].values
                    reg = LinearRegression().fit(X, Y)
                    slope = reg.coef_[0]
                    intercept = reg.intercept_
                    reg_line = slope * data[x_axis] + intercept
                    sns.lineplot(x=data[x_axis], y=reg_line, color='red', ax=ax)
                    
                    # Display regression equation
                    st.write(f"**Regression equation**: y = {slope:.2f}x + {intercept:.2f}")
                    st.write(f"**Slope**: {slope:.2f}")
                    st.write(f"**Intercept**: {intercept:.2f}")
            
            st.pyplot(fig)
