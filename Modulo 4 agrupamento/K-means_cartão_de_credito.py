import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = pd.read_csv('data/creditcard.csv')
