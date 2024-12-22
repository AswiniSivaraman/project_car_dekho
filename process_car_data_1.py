# ---------------------------------------------------------------   Data preprocessing  -------------------------------------------------------------------------------------------------------------------------
""" this file contains code which will flattened the unflattened columns in the dataset and combine them for all cities and merge all the city data into a single file 
and named the file as "Merged_Cities_data.csv" """


#import required modules
import pandas as pd
import numpy as np 
import ast

# file paths
bangalore_cars = "Datasets/bangalore_cars.xlsx"
chennai_cars = "Datasets/chennai_cars.xlsx"
delhi_cars = "Datasets/delhi_cars.xlsx"
hyderabad_cars = "Datasets/hyderabad_cars.xlsx"
jaipur_cars = "Datasets/jaipur_cars.xlsx"
kolkata_cars = "Datasets/kolkata_cars.xlsx"


# load datasets
bangalore_dataset = pd.read_excel(bangalore_cars)
chennai_dataset = pd.read_excel(chennai_cars)
delhi_dataset = pd.read_excel(delhi_cars)
hyderabad_dataset = pd.read_excel(hyderabad_cars)
jaipur_dataset = pd.read_excel(jaipur_cars)
kolkata_dataset = pd.read_excel(kolkata_cars)


# function to dispaly info about the dataframe
def dataframe_info(df):
    try:
        df.info()
    except Exception as e:
        print("Error occured when fetching the info of the dataset",e)


# function to convert string type dictioanries to python dictionaries and normalize them
def convert_strdict_to_pydict(df,col_name):
    try:
        df[col_name] = df[col_name].apply(ast.literal_eval)
        df_expanded = pd.json_normalize(df[col_name])
        return df_expanded
    except Exception as e:
        print("Error occurend when converting string type dictionary into python dictionary",e)
        return pd.DataFrame()

# flatten the new_car_overview column
def flatten_new_car_overview(x):
    try:
        overview_dict = ast.literal_eval(x)
        flattened_data = {}
        flattened_data['heading'] = overview_dict['heading']
        
        for i in overview_dict['top']:
            flattened_data[f"{i['key']}"] = i['value']
            flattened_data[f"{i['key']} Icon"] = i['icon']

        flattened_data['bottomData'] = np.nan if overview_dict['bottomData'] is None else overview_dict['bottomData']
        
        return flattened_data 
    
    except Exception as e:
        print("Error occured when flattening the new_car_overview column",e)
        return {}
    

# flatten_new_car_feature column
def flatten_new_car_feature(y):
    try:
        feature_dict = ast.literal_eval(y)
        flattened_data = {}
        flattened_data['heading'] = feature_dict['heading']
        
        for i, feature in enumerate(feature_dict['top']):
            flattened_data[f'top feature {i + 1}'] = feature['value']

        for category in feature_dict['data']:
            flattened_data[category['heading']] = len(category['list'])

        flattened_data['commonIcon'] = feature_dict['commonIcon']

        return flattened_data
                
    except Exception as e:
        print("Error occured when flattening the new_car_feature column:",e)
        return{}

# flatten the new_car_specs column
def flatten_new_car_specs(z):
    try:
        specs_dict = ast.literal_eval(z)
        flattened_data = {}

        flattened_data['heading'] = specs_dict['heading']

        for i in specs_dict['top']:
            flattened_data[f"{i['key']}"] = i['value']

        for j in specs_dict['data']:
            for k in j['list']:
                flattened_data[f"{k['key']}"] = k['value']

        flattened_data['commonIcon'] = specs_dict['commonIcon']

        return flattened_data
    
    except Exception as e:
        print("Error occured when flattening the new_car_specs column:",e)
        return {}
    
# concatenate columns for final output
def concat_columns(car_detail, car_overview, car_feature, car_specs, car_link):
    try:
        combined_data = pd.concat([car_detail.reset_index(drop=True),
                            car_overview.reset_index(drop=True),
                            car_feature.reset_index(drop=True),
                            car_specs.reset_index(drop=True), 
                            car_link.reset_index(drop=True)], axis=1)
        
        print("Columns concatenated successfully")
        return combined_data
        
    except Exception as e:
        print("Error occured when concatenating the columns", e)
        return pd.DataFrame()
    

# general processing function for all datasets
def process_city_dataset(dataset, city_name, column_1, column_2, column_3, column_4):
    try:
        car_detail = convert_strdict_to_pydict(dataset, column_1)
        print(f"Data processed successfully for '{column_1}' from the '{city_name}' dataset")
        # print(car_detail.head(5).to_string())

        dataset['new_car_overview_flattened'] = dataset[column_2].apply(lambda x: flatten_new_car_overview(x))
        car_overview = pd.json_normalize(dataset['new_car_overview_flattened'])
        print(f"Data flattened successfully for '{column_2}' from the '{city_name}' dataset")
        # print(car_overview.head(5).to_string())

        dataset['new_car_feature_flattened'] = dataset[column_3].apply(lambda y: flatten_new_car_feature(y))
        car_feature = pd.json_normalize(dataset['new_car_feature_flattened'])
        print(f"Data flattened successfully for '{column_3}' from the '{city_name}' dataset")
        # print(car_feature.head(5).to_string())

        dataset['new_car_feature_specs'] = dataset[column_4].apply(lambda z: flatten_new_car_specs(z))
        car_specs = pd.json_normalize(dataset['new_car_feature_specs'])
        print(f"Data flattened successfully for '{column_4}' from the '{city_name}' dataset")
        # print(car_specs.head(5).to_string())

        car_links = dataset['car_links']

        flattened_data = concat_columns(car_detail, car_overview, car_feature, car_specs, car_links)
        flattened_data['city'] = city_name

        return flattened_data
    
    except Exception as e:
        print(f'Error occured when processing the {city_name} dataset:',e)
        return {}
    
# combine data from all files and merge it into a single file
def combine_files():
    try:

        file_names = ['Flattened_dataset/bangalore.csv', 'Flattened_dataset/chennai.csv',
                      'Flattened_dataset/delhi.csv', 'Flattened_dataset/hyderabad.csv',
                      'Flattened_dataset/jaipur.csv', 'Flattened_dataset/kolkata.csv']
        
        data_list = []

        for file in file_names:
            df = pd.read_csv(file)
            data_list.append(df)

        combined_cities = pd.concat(data_list, ignore_index=True)
        combined_cities.to_csv('Merged_Cities_data.csv', index= False)

        print("All files succcessfully combined and written to 'Merged_Cities_data.csv'")
  
    except Exception as e:
        print("Error occured when combining the flattened city files:",e)


# column names
column_one = 'new_car_detail'
column_two = 'new_car_overview'
column_three = 'new_car_feature'
column_four = 'new_car_specs'

# process all city datasets
banglore_flattened_data = process_city_dataset(bangalore_dataset, 'bangalore', column_one, column_two, column_three, column_four)
chennai_flattened_data = process_city_dataset(chennai_dataset, 'chennai', column_one, column_two, column_three, column_four)
delhi_flattened_data = process_city_dataset(delhi_dataset, 'delhi', column_one, column_two, column_three, column_four)
hyderabad_flattened_data = process_city_dataset(hyderabad_dataset, 'hyderabad', column_one, column_two, column_three, column_four)
jaipur_flattened_data = process_city_dataset(jaipur_dataset, 'jaipur', column_one, column_two, column_three, column_four)
kolkata_flattened_data = process_city_dataset(kolkata_dataset, 'kolkata', column_one, column_two, column_three, column_four)


# Reset the index for each city DataFrame
banglore_flattened_data.reset_index(drop=True, inplace=True)
chennai_flattened_data.reset_index(drop=True, inplace=True)
delhi_flattened_data.reset_index(drop=True, inplace=True)
hyderabad_flattened_data.reset_index(drop=True, inplace=True)
jaipur_flattened_data.reset_index(drop=True, inplace=True)
kolkata_flattened_data.reset_index(drop=True, inplace=True)


# testing command out later
banglore_flattened_data.to_csv('Flattened_dataset/bangalore.csv', index=False)
chennai_flattened_data.to_csv('Flattened_dataset/chennai.csv', index=False)
delhi_flattened_data.to_csv('Flattened_dataset/delhi.csv', index=False)
hyderabad_flattened_data.to_csv('Flattened_dataset/hyderabad.csv', index=False)
jaipur_flattened_data.to_csv('Flattened_dataset/jaipur.csv', index=False)
kolkata_flattened_data.to_csv('Flattened_dataset/kolkata.csv', index=False)

# calling the function
combine_files()

