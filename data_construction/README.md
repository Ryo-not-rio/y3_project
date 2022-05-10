# Code for processing the scraped data into formats that can be used
The scraped data is cleaned and formatted using the code in this directory to be stored in root/data/parsed_data or root/data/parsed_data2. Also contains some data analysis code for the dataset.

# Instructions for data construction
1. Ensure both MD&A and financials have been scraped into ../data using the code provided in ../data_gathering
2. Run data_parser.py

# Instructions for data analysis
1. Ensure the data have been scraped into ../data
2. Ensure train_test_split_raw.csv exist in ../models
3. Run the functions in ./data_analysis.py depending on what analysis is required. 

