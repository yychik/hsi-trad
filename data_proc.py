
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import *


# In[2]:

def generate_sql(start_date, end_date, table, columns='*'):
    
    #-----------------------------------------------------#
    # Function to generate sql query statement for a table and specific list of columns over time
    #
    # INPUTS:
    # start_date = start date of the query (datetime)
    # end_date = end date of the query (datetime)
    # table = name of table in database (string)
    # column = field names to query (list)
    #
    # OUTPUTS:
    # sql = sql query statement (string)
    #-----------------------------------------------------#
    
    #Check format
    assert isinstance(start_date, datetime.date), 'Start Date has to be datetime.date object!'
    assert isinstance(end_date, datetime.date), 'End Date has to be datetime.date object!'
    
    #Turn into ISO format
    sd = start_date.isoformat()
    ed = end_date.isoformat()
    
    #Collapse columns into string
    c = ','.join(columns)
        
    #Generate SQL statement
    sql = 'SELECT ' + c + ' from HSI_DATA.' + table + ' WHERE DATE between ' + '\'' + sd + '\'' + ' AND ' + '\'' + ed + '\''

    return sql


# In[1]:

def get_data(conn, start_date, end_date, table, columns):
    
    #-----------------------------------------------------#
    # Function to get data from mysql server
    #
    # INPUTS:
    # conn = mysqlclient connection (mysqlclient conn object)
    # start_date = start date of the query (datetime)
    # end_date = end date of the query (datetime)
    # table = name of table in database (string)
    # column = field names to query (list)
    #
    # OUTPUTS:
    # data = pandas DataFrame
    #-----------------------------------------------------#
    
    #Get data
    data = pd.read_sql(generate_sql(start_date, end_date, table, columns), con=conn)
    
    #Set index if Date is contained in the dataframe
    if 'DATE' in list(data.keys()):
        data['DATE'] = pd.to_datetime(data['DATE'])
        data.set_index(['DATE'], inplace=True)
    
    return data


# In[4]:

def get_fwd_ret(conn, start_date, end_date, step):
    
    #-----------------------------------------------------#
    # Function to generate forward returns of tracker fund for label generation
    #
    # INPUTS:
    # conn = mysqlclient connection (mysqlclient conn object)
    # start_date = start date of the query (datetime)
    # end_date = end date of the query (datetime)
    # step = forward step-day return (int)
    #
    # OUTPUTS:
    # data = forward returns (pandas DataFrame)
    #-----------------------------------------------------#
    
    #Check format
    assert isinstance(start_date, datetime.date), 'Start Date has to be datetime.date object!'
    assert isinstance(end_date, datetime.date), 'End Date has to be datetime.date object!'
    
    #Get data
    data_now = get_data(conn, start_date, end_date, 'hsi_data', ['DATE', 'CLOSE'])
    
    #First date from current data
    first_date = data_now.index[0]
    last_date = data_now.index[-1]
    
    #Get next dates
    fwd_startdate_sql = 'SELECT DATE FROM HSI_DATA.hsi_data WHERE DATE > ' + '\'' + first_date.isoformat() + '\'' + ' LIMIT 1'
    fwd_enddate_sql = 'SELECT DATE FROM HSI_DATA.hsi_data WHERE DATE > ' + '\'' + last_date.isoformat() + '\'' + ' LIMIT 1'
        
    #Get dates
    fwd_startdate = pd.read_sql(fwd_startdate_sql, con=conn).get_values()[0,0]
    fwd_enddate = pd.read_sql(fwd_enddate_sql, con=conn).get_values()[0,0]
    
    #print(fwd_startdate)
    #print(fwd_enddate)
    
    #Fetch data for current
    data_fwd = get_data(conn, fwd_startdate, fwd_enddate, 'hsi_data', ['CLOSE'])
    data_fwd.set_index(data_now.index, inplace=True)
        
    #Calculate forward returns
    data = np.log(data_fwd/data_now)
    data.columns = ['FWD_RET']
    
    return data
    


# In[5]:

def get_labels_from_fwd_ret(fwd_ret, upper_cutoff, lower_cutoff):
    
    #-----------------------------------------------------#
    # Function to generate classification labels from forward returns for prediction
    # 
    # INPUTS:
    # fwd_ret: dataframe of forward returns (pandas dataframe)
    # upper_cutoff: cutoff % for the classification boundary for "up" label
    # lower_cutoff: cutoff % for classification boundary for "down" label
    #
    # OUTPUT:
    # labels: dataframe of class labels (pandas dataframe)
    #-----------------------------------------------------#
    
    #Classify the returns
    labels_array = [0 if ret < lower_cutoff else 2 if ret > upper_cutoff else 0 for ret in fwd_ret.values]
    
    labels = pd.DataFrame(labels_array).set_index(fwd_ret.index)
    labels.columns = ['LABELS']
    
    return labels


# In[ ]:

def train_val_test_split(inputs, train_start, val_start, test_start, test_end, normalize='RobustScaler'):
    
    #-----------------------------------------------------#
    # Function to split data into training, validation and test set by date.
    # This will generate overlapping samples of a step sie feed into LSTM model
    #
    # INPUTS:
    # inputs: pandas array of data to be split, indexed by time
    # train_start: validation set start date
    # val_start: test set start date
    # test_start: test set start date
    # normalize: parameter to control what sklearn normalization to use. It takes whatever sklearn is providing, or None
    #
    # OUTPUT:
    # scaler, (inputs_train, inputs_val, inputs_test): sklearn scaler object, tuples of numpy array
    #-----------------------------------------------------#
    
    #Split the data
    inputs_train = inputs[train_start:val_start]
    inputs_val = inputs[val_start:test_start]
    inputs_test = inputs[test_start:test_end]
    
    #Define dictionary to store the corresponding mapping between normalize argument and its scaler object
    dict_scaler = {'MinMaxScaler': MinMaxScaler(), 
                   'MaxAbsScaler': MaxAbsScaler(),
                   'StandardScaler': StandardScaler(),
                   'RobustScaler': RobustScaler(),
                   'Normalizer': Normalizer()}

    
    #Normalize the data
    if normalize != None:
        
        #Define and fit
        scaler = dict_scaler[normalize]
        scaler.fit(inputs_train)
        
        #Transform
        inputs_train, inputs_val, inputs_test = scaler.transform(inputs_train), scaler.transform(inputs_val), scaler.transform(inputs_test)
        
        #Output
        return scaler, inputs_train, inputs_val, inputs_test
    
    else:
        
        #Output
        return None, inputs_train.as_matrix(), inputs_val.as_matrix(), inputs_test.as_matrix()
        


# In[1]:

def get_batch(inputs, labels, batch_size, steps):
    
    #-----------------------------------------------------#
    # Function to generate batches inputs to train LSTM model.
    # This will generate overlapping samples of a step sie feed into LSTM model
    #
    # INPUTS:
    # inputs: numpy array of data to be batched
    # batch_size: size of each batch
    # steps: step size to feed into model
    #
    # OUTPUT:
    # batch_data: numpy array of (input_batch, labels_batch) tuple
    #-----------------------------------------------------#
        
    #Calculate number of sequences with each "steps" step size able to generate
    n_seq = inputs.shape[0] - steps + 1
    
    #Calculate number of batches
    n_samples = n_seq * steps
    n_batches = n_samples // (batch_size * steps) # 1 more batch to capture the residual data
    
    #Error Check: Assert n_batches has to be > 0
    assert(n_batches > 0), 'Not enough data to form 1 batch!'
    
    #labels
    #labels_batch = labels[:n_batches * batch_size, :]
    
    #Generate batches
    for n in range(0, n_batches):
        
        #Container to store the outputs
        inputs_batch = []
        labels_batch = []
        
        for ii in range(n * batch_size, (n + 1) * batch_size):
                        
            inputs_batch.append(inputs[ii : ii + steps, :])
            labels_batch.append(labels[ii : ii + steps])
        
        #Return batches
        yield np.stack(inputs_batch), np.stack(labels_batch)


# In[ ]:



