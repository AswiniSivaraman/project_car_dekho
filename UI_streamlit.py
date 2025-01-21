# Import required modules
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

if 'reset' not in st.session_state:
    st.session_state['reset'] = False

# Read the file and save the data in a variable called df
df = pd.read_csv('final_data.csv')

categorical_columns = ['oem', 'model', 'Tyre Type']
unique_values = {col: df[col].unique().tolist() for col in categorical_columns}

brand_model_mapping = df.groupby('oem')['model'].unique().to_dict()

model = joblib.load('carprice_prediction_ml_model.pkl')
encoded_mappings = joblib.load("encoded_mappings.pkl")

def predict_price(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Setting the page configuration data
st.set_page_config(
    page_title="Car Dekho",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="auto"
)

# Sidebar styling
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

# Options styling in sidebar and added image in sidebar
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

# Home section
if selected == "Home":

    st.title(":blue[Car Dekho] - A Used Car Price Prediction App 🚙")

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

    st.subheader("Welcome to the Used Car Price Prediction App!")

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


# Car Price Prediction section
if selected == "Car Price Predict":

    st.title(":blue[Car Price Prediction] - 🔍 :blue[Let's Predict The Used Car Price]")

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if st.session_state.reset:
        st.session_state.company = "--- select the Company name ---"
        st.session_state.model = "--- select the Model name ---"
        st.session_state.tyre_type = "--- select the Tyre Type name ---"
        st.session_state.reset = False

    company = st.selectbox('Company', ['--- select the Company name ---'] + unique_values['oem'], index=0, key="company")
    
    

    if company != '--- select the Company name ---':

        selected_model = st.selectbox('Model', brand_model_mapping.get(company, []), index=0)
        tyre_type = st.selectbox('Tyre Type', ['--- select the Tyre Type name ---'] + unique_values['Tyre Type'], index=0, key="tyre_type")
        registration_year = st.number_input('Registration Year', min_value=int(df['Registration Year'].min()), max_value=int(df['Registration Year'].max()), value=2022)
        safety = st.number_input('Safety', min_value=int(df['Safety'].min()), max_value=int(df['Safety'].max()), value=3)
        max_power = st.number_input('Max Power (bhp)', min_value=int(df['Max Power'].min()), max_value=int(df['Max Power'].max()), value=100)
        width = st.number_input('Width (mm)', min_value=int(df['Width'].min()), max_value=int(df['Width'].max()), value=1500)
        wheel_base = st.number_input('Wheel Base (mm)', min_value=int(df['Wheel Base'].min()), max_value=int(df['Wheel Base'].max()), value=2500)
        gear_box = st.number_input('Gear Box', min_value=int(df['Gear Box'].min()), max_value=int(df['Gear Box'].max()), value=5)
        turning_radius = st.number_input('Turning Radius (m)', min_value=float(df['Turning Radius'].min()), max_value=float(df['Turning Radius'].max()), value=5.0)
        acceleration = st.number_input('Acceleration (0-100 km/h in sec)', min_value=float(df['Acceleration'].min()), max_value=float(df['Acceleration'].max()), value=10.0)
        wheel_size = st.number_input('Wheel Size (inches)', min_value=int(df['Wheel Size'].min()), max_value=int(df['Wheel Size'].max()), value=16)
        owner_no = st.number_input('Owner Number', min_value=int(df['ownerNo'].min()), max_value=int(df['ownerNo'].max()), value=1)
        model_year = st.number_input('Model Year', min_value=int(df['modelYear'].min()), max_value=int(df['modelYear'].max()), value=2022)

        input_data = pd.DataFrame({
            'Registration Year': [registration_year],
            'Safety': [safety],
            'Max Power': [max_power],
            'Width': [width],
            'Wheel Base': [wheel_base],
            'Gear Box': [gear_box],
            'Turning Radius': [turning_radius],
            'Acceleration': [acceleration],
            'Tyre Type': [encoded_mappings['Tyre Type'][tyre_type]],
            'Wheel Size': [wheel_size],
            'oem': [encoded_mappings['oem'][company]],
            'model': [encoded_mappings['model'][selected_model]],
            'ownerNo': [owner_no],
            'modelYear': [model_year]
        })

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Predict"):
                predicted_price = predict_price(input_data)
                formatted_price = f"₹ {predicted_price:,.2f}"
                st.write(f"<p style='font-size: 40px; font-weight: bold; color: #0492C2; text-align: center;'>Predicted Price: {formatted_price}</p>", unsafe_allow_html=True)

        with col2:
            if st.button("Reset"):
                st.session_state.reset = True
                st.experimental_rerun()

    else:
        st.markdown("<p style='color:blue; font-size:20px;'>Please make valid selections for all dropdowns to proceed.</p>", unsafe_allow_html=True)
