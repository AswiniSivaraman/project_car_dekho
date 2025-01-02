# Import required modules
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

if 'reset' not in st.session_state:
    st.session_state['reset'] = False

# read the file and save the data in variable called df
df = pd.read_csv('domain_related_data.csv')

categorical_columns = ['oem', 'transmission', 'Fuel Type', 'model', 'bt']
unique_values = {col: df[col].unique().tolist() for col in categorical_columns}

brand_model_mapping = df.groupby('oem')['model'].unique().to_dict()

model = joblib.load('carprice_prediction_ml_model.pkl')
encoded_mappings = joblib.load("encoded_mappings.pkl")

def predict_price(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# setting the page configuration data
st.set_page_config(
    page_title="Car Dekho",
    page_icon=":car:",
    layout="wide",      #if wide full page will cover , centered means centre part of the page is covered
    initial_sidebar_state="auto"
    )

#sidebar styling 
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #FFFFFF;
        margin-right: 20px;
        border-right: 2px solid #FFFFFF
    }
</style>
""", unsafe_allow_html=True)

data = r'images\car_logo.jpg'

#options styling in sidebar and added image in sidebar
with st.sidebar:

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%; /* Adjust as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Car Price Predict"],
        icons=["house-door-fill", "search"],
        menu_icon="car", 
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#FAF9F6"},
            "icon": {"color": "#000000", "font-size": "23px"},
            "nav-link": {"font-size": "16px","text-align": "left","margin": "0px","--hover-color": "#abdbf7","font-weight": "bold"},
            "nav-link-selected": {"background-color": "#0f85ca","color": "white","font-weight": "bold"},
        },
    )

    st.sidebar.image(data, use_column_width=False)

# if selected option is home means then add these in the page
if selected == "Home":

    st.title(":blue[Car Dekho] - A Used Car Price Prediction App 🚙")

    # Load custom CSS for styling if available
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add custom CSS for heading colors
    st.markdown("""
    <style>
        h3 {
            color: blue; 
        }
        h2 {
            color: blue; 
        }
        h1 {
            color: blue; 
        }
    </style>
    """, unsafe_allow_html=True)

    # Subheader
    st.subheader("Welcome to the Used Car Price Prediction App!")

    # Main content
    st.markdown("""
    ### **About the App**
    Are you planning to buy or sell a used car and wondering what the fair price might be? Look no further! 
    Our **Car Dekho** uses advanced machine learning algorithms to predict the market value of a used car based on various factors like brand, model, year of manufacture, mileage, fuel type, and more.
    
    ### **Why Use This App?**
    1. **Accurate Predictions**: Get reliable price estimates tailored to your car's specifications.
    2. **Time-Saving**: No need to manually browse through endless listings or consult multiple dealers.
    3. **Informed Decisions**: Make smarter buying and selling decisions with data-driven insights.
    4. **User-Friendly**: A simple and intuitive interface designed for everyone.
    
    ### **How Does It Work?**
    1. **Input Details**: Enter your car's details such as make, model, year, mileage, fuel type, and other key specifications.
    2. **Analyze the Data**: Our app processes your inputs using cutting-edge predictive models.
    3. **Get the Price**: Instantly receive an estimated market value for the car.
    
    ### **Who Can Use This App?**
    - **Car Buyers**: Find out if the price of a used car you’re interested in is fair.
    - **Car Sellers**: Determine the best selling price for your car to maximize your profit.
    - **Car Dealers**: Use this app to streamline your pricing strategies and attract more customers.
    
    ### **Ready to Begin?**
    Head over to the **Car Price Predict** section to start predicting the value of your car now! 🚘
    """)

# ['ownerNo', 'km', 'modelYear', 'Engine Displacement', 'Mileage','Max Power', 'Gear Box', 'price']

if selected=="Car Price Predict":

    st.title(":blue[Car Price Prediction] - 🔍 :blue[Let's Predict The Used Car Price]")

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add custom CSS for input elements with borders
    st.markdown("""
    <style>
        /* Add border for dropdowns */
        div[data-baseweb="select"] {
            border: 2px solid #007bff !important; /* Blue border */
            border-radius: 8px !important; /* Rounded corners */
            padding: 5px !important; /* Padding for better spacing */
        }
        div[data-baseweb="select"]:focus-within {
            border: 2px solid #0056b3 !important; /* Darker blue border on focus */
        }

        /* Add border for number input */
        input[type="number"] {
            border: 2px solid #007bff !important; /* Blue border */
            border-radius: 8px !important; /* Rounded corners */
            padding: 5px !important; /* Padding for better spacing */
        }
        input[type="number"]:focus {
            border: 2px solid #0056b3 !important; /* Darker blue border on focus */
        }

        /* General styling for all inputs */
        input[type="text"], textarea {
            border: 2px solid #007bff !important; /* Blue border */
            border-radius: 8px !important; /* Rounded corners */
            padding: 5px !important; /* Padding for better spacing */
        }
        input[type="text"]:focus, textarea:focus {
            border: 2px solid #0056b3 !important; /* Darker blue border on focus */
        }
    </style>
    """, unsafe_allow_html=True)

    # Add custom CSS for labels
    st.markdown("""
    <style>
        label {
            color: blue !important; /* Change text color to blue */
            font-weight: bold !important; /* Make text bold */
            font-size: 16px; /* Optional: Increase label size slightly */
        }
    </style>
    """, unsafe_allow_html=True)

    # Check reset flag to reset widget values
    if st.session_state.reset:
        st.session_state.company = "--- select the Company name ---"
        st.session_state.transmission = "--- select the Transmission name ---"
        st.session_state.fuel_type = "--- select the Fuel Type name ---"
        st.session_state.body_type = "--- select the Body Type name ---"
        st.session_state.reset = False

    # Dropdowns with placeholders
    company = st.selectbox('Company', ['--- select the Company name ---'] + unique_values['oem'], index=0, key="company")
    transmission = st.selectbox('Transmission', ['--- select the Transmission name ---'] + unique_values['transmission'], index=0, key="transmission")
    fuel_type = st.selectbox('Fuel Type', ['--- select the Fuel Type name ---'] + unique_values['Fuel Type'], index=0, key="fuel_type")
    body_type = st.selectbox('Body Type', ['--- select the Body Type name ---'] + unique_values['bt'], index=0, key="body_type")

    # Handle inputs only if valid selections are made
    if (company != '--- select the Company name ---' and transmission != '--- select the Transmission name ---' and fuel_type != '--- select the Fuel Type name ---' and body_type != '--- select the Body Type name ---'):

        selected_model = st.selectbox('Model', brand_model_mapping.get(company, []), index=0)

        owner_no = st.number_input('Owner Number', min_value=int(df['ownerNo'].min()), max_value=int(df['ownerNo'].max()), value=1)
        km_driven = st.number_input('Kilometers Driven', min_value=int(df['km'].min()), max_value=int(df['km'].max()), value=10000)
        model_year = st.number_input('Model Year', min_value=int(df['modelYear'].min()), max_value=int(df['modelYear'].max()), value=2022)
        engine_cc = st.number_input('Engine Displacement (CC)', min_value=int(df['Engine Displacement'].min()), max_value=int(df['Engine Displacement'].max()), value=1000)
        mileage = st.number_input('Mileage (kmpl)', min_value=float(df['Mileage'].min()), max_value=float(df['Mileage'].max()), value=15.0)
        max_power = st.number_input('Max Power (bhp)', min_value=int(df['Max Power'].min()), max_value=int(df['Max Power'].max()), value=100)
        gear_box = st.number_input('Gear Box', min_value=int(df['Gear Box'].min()), max_value=int(df['Gear Box'].max()), value=5)

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'oem': [encoded_mappings['oem'][company]],
            'ownerNo': [owner_no],
            'transmission': [encoded_mappings['transmission'][transmission]],
            'km': [km_driven],
            'modelYear': [model_year],
            'Fuel Type': [encoded_mappings['Fuel Type'][fuel_type]],
            'Engine Displacement': [engine_cc],
            'Mileage': [mileage],
            'Max Power': [max_power],
            'model': [encoded_mappings['model'][selected_model]],
            'Gear Box': [gear_box],
            'bt': [encoded_mappings['bt'][body_type]],
        })

        # Add custom CSS for button styling
        st.markdown("""
            <style>
            .stButton > button {
                font-size: 22px; /* Increase button font size */
                padding: 20px 90px; /* Increase button size */
                background-color: white; /* Default background color */
                color: black !important; /* Force text color to always be black */
                border: 2px solid #007bff; /* Blue border */
                border-radius: 8px; /* Rounded edges */
                font-weight: bold; /* Bold text */
                cursor: pointer;
            }
            .stButton > button:focus {
                background-color: #007bff !important; /* Change background to blue when clicked */
                color: black !important; /* Ensure text color stays black */
            }
            .stButton > button:hover {
                background-color: #e7f3ff; /* Optional: Light blue hover effect */
                color: black !important; /* Ensure text color stays black on hover */
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Predict Button
        with col1:
            if st.button("Predict"):
                predicted_price = predict_price(input_data)
                formatted_price = f"₹ {predicted_price:,.2f}"
                st.write("<p style='font-size: 48px; font-weight: bold; color: #0492C2; text-align: center;'>Predicted Price: {}</p>".format(formatted_price), unsafe_allow_html=True)
        
        # Reset Button
        with col2:
            if st.button("Reset"):
                st.session_state.reset = True
                st.experimental_rerun()

    else:
        st.markdown("<p style='color:blue; font-size:20px;'>Please make valid selections for all dropdowns to proceed.</p>",unsafe_allow_html=True)





