import numpy as np 

import pandas as pd
def resize_data(dataset): 

    dataset.replace(' NA', -99, inplace=True)

    dataset.fillna(-99, inplace=True)

    

    for col in list(dataset.columns):

        if dataset[col].dtype == 'int64' or dataset[col].dtype == 'float64':

            dataset[col] = dataset[col].astype(np.int8)    

                

    return dataset
reader = pd.read_csv('../input/train_ver2.csv', chunksize=10000)

df = pd.concat([resize_data(chunk) for chunk in reader])
{

  "cells": [

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "a21614ae-0eff-6d62-0d11-d7701efeaac0"

      },

      "outputs": [],

      "source": [

        "import numpy as np \n",

        "import pandas as pd"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "e1316861-7944-823d-5b56-9b8627dc67d9"

      },

      "outputs": [],

      "source": [

        "def resize_data(dataset): \n",

        "    dataset.replace(' NA', -99, inplace=True)\n",

        "    dataset.fillna(-99, inplace=True)\n",

        "    \n",

        "    for col in list(dataset.columns):\n",

        "        if dataset[col].dtype == 'int64' or dataset[col].dtype == 'float64':\n",

        "            dataset[col] = dataset[col].astype(np.int8)    \n",

        "                \n",

        "    return dataset"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "3cecbdc9-d355-8c6b-6390-3e7c973e5869"

      },

      "outputs": [],

      "source": [

        "reader = pd.read_csv('../input/train_ver2.csv', chunksize=10000)\n",

        "df = pd.concat([resize_data(chunk) for chunk in reader])"

      ]

    },

    {

      "cell_type": "markdown",

      "metadata": {

        "_cell_guid": "4464968e-6a3f-7ec8-ad8e-14754730fab6"

      },

      "source": [

        "Now train dataframe usage 2.1 Gb memory"

      ]

    }

  ],

  "metadata": {

    "_change_revision": 0,

    "_is_fork": false,

    "kernelspec": {

      "display_name": "Python 3",

      "language": "python",

      "name": "python3"

    },

    "language_info": {

      "codemirror_mode": {

        "name": "ipython",

        "version": 3

      },

      "file_extension": ".py",

      "mimetype": "text/x-python",

      "name": "python",

      "nbconvert_exporter": "python",

      "pygments_lexer": "ipython3",

      "version": "3.5.2"

    }

  },

  "nbformat": 4,

  "nbformat_minor": 0

}