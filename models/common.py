'''
This file implements functions necessary for both the baseline model and the proposed model.
'''
import os
import pickle
from sklearn import preprocessing
import h5py
from numpy import nan
import random
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.realpath(__file__))

def get_fin_features(raw_data):
    # The features to be used from the financial data
    fin_features = ['Accounts Payable', 'Accounts Receivable', 'Acquisitions Net', 'CAPEX', 'COGS', 'COGS and Expenses',
                    'Cash Provided by Operating Activities', 'Cash Used for Investing Activites',
                    'Cash Used/Provided by Financing Activities', 'Cash and Cash Equivalents',
                    'Cash and Short-Term Investments', 'Cash at the Beginning of Period', 'Cash at the End of Period',
                    'Change in Working Capital', 'Common Stock', 'Common Stock Issued', 'Common Stock Repurchased',
                    'Debt Repayment', 'Deferred Income Tax', 'Deferred Revenue', 'Deferred Tax Liabilities',
                    'Depreciation and Amortization', 'Dividends Paid', 'EBITDA', 'EBITDA ratio', 'EPS', 'EPS Diluted',
                    'Effect of Forex Changes on Cash', 'Free Cash Flow', 'General and Administrative Exp.', 'Goodwill',
                    'Goodwill and Intangible Assets', 'Gross Profit', 'Gross Profit ratio', 'Income Before Tax',
                    'Income Before Tax ratio', 'Income Tax expense', 'Intangible Assets', 'Interest Expenese',
                    'Inventory',
                    'Investments', 'Long-Term Debt', 'Net Change In Cash', 'Net Income', 'Net Income ratio',
                    'Net Receivables', 'Operating Expenses', 'Operating Income', 'Operating Income ratio',
                    'Other Assets',
                    'Other Comprehensive Income/Loss', 'Other Current Assets', 'Other Current Liabilities',
                    'Other Expenses', 'Other Financing Activites', 'Other Investing Activites', 'Other Liabilities',
                    'Other Non-Cash Items', 'Other Non-Current Assets', 'Other Non-Current Liabilities',
                    'Other Total Stockholders Equity', 'Other Working Capital', 'PP&E', 'Purchases of Investments',
                    'Research and Development Exp.', 'Retained Earnings', 'Revenue', 'Sales/Maturities of Investments',
                    'Selling and Marketing Exp.', 'Selling, General and Administrative Exp.', 'Short-Term Debt',
                    'Short-Term Investments', 'Stock Based Compensation', 'Tax Assets', 'Tax Payable', 'Total Assets',
                    'Total Current Assets', 'Total Current Liabilities', 'Total Liabilities',
                    'Total Liabilities And Stockholders Equity', 'Total Non-Current Assets',
                    'Total Non-Current Liabilities', 'Total Other Income Expenses Net', 'Total Stockholders Equity',
                    'Weighted Average Shares Outstanding', 'Weighted Average Shares Outstanding Diluted']
    if raw_data:
        fin_features.append('market capitalization')
    return fin_features


'''
Return the normalizer object fitted on the financial data.

The normalizer is a StandardScaler provided by sklearn and 
is fitted on the three years of financials but not on the market average financials.

@param force_new :: If True, a new normalizer is created. If False, an existing one is returned, given one exists.
'''

def get_normalizer(raw_data=False, force_new=False):
    fin_features = get_fin_features(raw_data)
    data_path = os.path.join("..", "data", "parsed_data") if raw_data else os.path.join("..", "data", "parsed_data2")
    file_name = "scaler_raw.pickle" if raw_data else "scaler.pickle"
    file_name = os.path.join(DATA_DIR, file_name)

    # Create new normalizer if file doesn't exist or force_new==True
    if not os.path.isfile(file_name) or force_new:
        print("Creating new scaler...")
        # For each data point add the financial data to a list
        datas = []
        for file in os.listdir(data_path):
            with open(os.path.join(data_path, file), "rb") as f:
                data = pickle.load(f)

            # For each financial year in the data point
            for key in ["y-2fin", "y-1fin", "y0fin"]:
                fin = data[key]
                # Sort the keys so all the features are placed in the correct column
                sorted_data = [fin[k] for k in fin_features]
                datas.append(sorted_data)

        scaler = preprocessing.StandardScaler()
        scaler.fit(datas)  # Fit the scaler onto the financial data
        with open(file_name, "wb") as f:
            pickle.dump(scaler, f)

    with open(file_name, "rb") as f:
        return pickle.load(f)


# Create a .hdf5 version of the whole dataset
def create_h5_dataset(raw_data=False):
    data_path = os.path.join("..", "data", "parsed_data") if raw_data else os.path.join("..", "data", "parsed_data2")
    filename = "parsed_data_raw.hdf5" if raw_data else "parsed_data.hdf5"
    filename = os.path.join(DATA_DIR, filename)
    if os.path.exists(filename):
        os.remove(filename)
    h5f = h5py.File(filename, "a")
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), "rb") as f:
            data = pickle.load(f)
        h5f.create_dataset(file, data=str(data))

    return h5f


# Load the .hdf5 dataset into memory
def load_h5_dataset(raw_data=False):
    filename = "parsed_data_raw.hdf5" if raw_data else "parsed_data.hdf5"
    filename = os.path.join(DATA_DIR, filename)
    datas = {}
    h5f = h5py.File(filename, "r")
    for key in h5f.keys():
        datas[key] = eval(h5f[key][()])
    return datas


'''
Return the file names of the files in the train set and test set.

The split is specified in train_test_split.csv. This function ensures the 
train-test split is the same for all models. Also this ensures each model does
not accidentally include a file used for training in their test set.

@param test_ratio :: the ratio of test data to train data.
'''
def train_test_files(raw_data=False,  test_ratio=0.2, force_new=False):
    filename = "train_test_split_raw.csv" if raw_data else "train_test_split.csv"
    filename = os.path.join(DATA_DIR, filename)
    data_dir = os.path.join("..", "data", "parsed_data") if raw_data else os.path.join("..", "data", "parsed_data2")
    # If file doesn't exist, create a new split
    if not os.path.isfile(filename) or force_new:
        files = os.listdir(data_dir)
        companies = defaultdict(list)
        for file in files:
            company = "".join(file.split("-")[:-1])
            companies[company].append(file)

        test_num = int(len(files) * test_ratio)
        count = 0
        test_files = []
        while count < test_num:
            company = random.choice(list(companies.keys()))
            test_files += companies[company]
            count += len(companies[company])
            del companies[company]
        train_files = [item for sublist in companies.values() for item in sublist]
        with open(filename, "w") as f:
            f.write(",".join(train_files) + "\n" + ",".join(test_files))
        return train_files, test_files

    # Load the split from file
    with open(filename) as f:
        train_str, test_str = f.readlines()
        train_str = train_str[:-1]

    train_files = train_str.split(",")
    test_files = test_str.split(",")
    return train_files, test_files


def prepare_data():
    create_h5_dataset(raw_data=True)
    get_normalizer(raw_data=True, force_new=True)
    train_test_files(raw_data=True, force_new=True)

if __name__ == "__main__":
    prepare_data()
    # get_normalizer(force_new=True)
    # create_h5_dataset()
    # load_h5_dataset()
    # train, test = train_test_files()
    # print(len(train), len(test))