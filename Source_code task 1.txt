###################: TASK 1 :##################
import pandas as pd
import requests
# Now we will call our API and send get request to the api
rspns=requests.get('https://api.binance.com/api/v3/ticker/24hr')
# Now we will check whether the request was successful or not
if rspns.status_code==200:
    data=rspns.json()
    dataframe=pd.DataFrame(data)
    # here we will create columns
    columns=['symbol','lastPrice','volume']
    # now we will filter the columns
    dataframe = dataframe[columns]
    # now we will save the data as csv file
    dataframe.to_csv('Binance_data.csv',index=False)
    print("CSV dataset has been created.")
else:
    print("Operation has been failed to fetch data from API. status code:", rspns.status_code)
