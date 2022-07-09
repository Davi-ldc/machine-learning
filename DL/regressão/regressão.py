#(ja ta explicado na pasta ML)
import numpy as np
import pandas as pd

data = pd.read_csv('dataDL/autos.csv', encoding='ISO-8859-1')

data.drop(['dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name'], axis=1)


