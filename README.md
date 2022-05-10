# Stock selection using MD&A and financials
This project attempts to create a model that is able to select out-performing stocks using MD&A and financials of a company as input features to a NN model. Most code is written in .py files apart from the code to execute training which is written in a .ipynb to be executed in Google Colaboratory.
The project also includes code to scrape the data necessary from the web and the code to construct the dataset from the scraped data. However, some of the scraped data has been removed. Constructed datasets however have been kept so the training code can be run.  

## Directory structure
- ./data - contains scraped data
- ./data_construction - contains code for data cleaning and dataset construction from scraped data
- ./data_gathering - contains code for data scraping
- ./evaluation - contains code used in the evaluation stage
- ./models - contains code for the implementation of both the baseline model and proposed NN model. Also includes some cached dataset, cached prediction results and saved models.

## Instructions for running each section of code
Each directory of code will include a README.md file which includes instructions for running that part of the code. For training the model, it is recommended to run the code in ./models in a local environment for simplicity. However, the whole folder containing this README.md file can be uploaded to google drive to run training.ipynb in Google Colaboratory. Evaluation should not be performed on Google Colab as loading the model from Google Drive is unstable and inconsistent. This issue does not occur in a local environment.

## Requirements
- Python >= 3.8
- Tensorflow >= 2.4
- Libraries listed in requirements.txt. Install these with `pip install -r requirements.txt`