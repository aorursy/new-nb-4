#https://www.kaggle.com/c/nyc-taxi-trip-duration/data

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/train.csv')
dg = pd.read_csv('../input/test.csv')
dh = pd.read_csv('../input/sample_submission.csv')
df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
dg.sample(5)
df.pickup_datetime = pd.to_datetime(df.pickup_datetime, format = '%Y-%m-%d %H:%M:%S')
dg.pickup_datetime = pd.to_datetime(dg.pickup_datetime, format = '%Y-%m-%d %H:%M:%S')
DF = df.set_index('pickup_datetime')
DF.trip_duration.plot.line()
# Can I trust those very large values?