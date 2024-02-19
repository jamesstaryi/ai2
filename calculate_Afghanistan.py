import pandas as pd
import numpy as np
'''Write your code here '''
df = pd.read_csv("Life Expectancy Data.csv")

'''end of student code'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
total_df = df[df.Country == 'Afghanistan'][['Year', 'GDP', 'Life expectancy ']]
for degree in [1,2,3,4]:
    #Step 1:You should define the train_x, each row of it represents a year of GDP of Afghanistan,
    #and each column of it represents a power of the GDP. The Year column should be used to select the samples.
    #Step 2:Define train_y, each row of it represents a year of Life expectancy of Afghanistan. The Year column should be used to select the samples.
    #Step 3:Define a LinearRegression model, and fit it using train_X and train_y.
    #Step 4:Calculate rmse and r2_score using the fitted model.
    '''Write your code here '''
    train_x = np.array([total_df[total_df.Year <= 2013].GDP**i
                        for i in range(1,degree+1)]).transpose(1,0).reshape((-1, degree))
    train_y = np.array(total_df[total_df.Year <= 2013]['Life expectancy '])

    model = LinearRegression()
    model.fit(train_x,train_y)

    rmse = np.sqrt(mean_squared_error(model.predict(train_x), train_y))
    r2_score = model.score(train_x, train_y)
    
    '''end of student code'''
    print(f'Train set, degree={degree}, RMSE={rmse:.3f}, R2={r2_score:.3f}')
    #feel free to change the variable name if needed. 
    #DO NOT change the output format.
    #Step 1: Define test_x and test_y by selecting the remaining years of the data
    #Step 2: Use model.predict to generate the prediction
    #Step 3: Calculate rmse and r2_score on test_x and test_y.
    '''Write your code here'''
    test_x = np.array([total_df[total_df.Year > 2013].GDP**i
                        for i in range(1,degree+1)]).transpose(1,0).reshape((-1, degree))
    test_y = np.array(total_df[total_df.Year > 2013]['Life expectancy '])

    predictions = model.predict(test_x)

    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2_score = model.score(test_x, test_y)
    
    '''end of student code'''
    print(f'Test set,  degree={degree}, RMSE={rmse:.3f}, R2={r2_score:.3f}')