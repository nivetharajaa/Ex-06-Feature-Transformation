# Ex-06-Feature-Transformation

##AIM

To read the given data and perform Feature Transformation process and save the data to a file.

##ALGORITHM

##STEP 1

Read the given Data

##STEP 2

Clean the Data Set using Data Cleaning Process

##STEP 3

Apply Feature Transformation techniques to all the feature of the data set

##STEP 4

Save the data to the file

CODE

import pandas as pd

df=pd.read_csv('/content/Data_to_Transform.csv')

df.head()

df.isnull().sum()

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer pt=PowerTransformer("yeo-johnson")

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

##OUTPUT

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/1ed3d57b-3354-4c71-8799-153d785aa246)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/14eee2bb-623f-47f9-946e-26bc7bf389d6)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/4296b790-33d1-4022-ba78-e8c279444650)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/e7e3e5d8-ac7e-4a3a-9eab-c779a16b920c)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/9f1cb717-4a76-42b3-98a1-f324b93bd69b)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/55598367-5b0e-46c0-97ca-2c1f3614180a)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/850fab3f-d05d-4afe-ba7c-39d6d6131f2d)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/c138626d-4a92-47bd-94d2-f1648864ce92)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/3b1ae232-6e38-4e1c-99cd-009722e89210)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/8143dedc-1c46-44b4-8d79-d14bae4e6f2c)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/d30026d5-47c1-481c-a715-1c88be8d4f4e)

![image](https://github.com/nivetharajaa/Ex-06-Feature-Transformation/assets/120543388/7d1ee3ea-96c3-4352-a2c3-42a35a293ec3)

##RESULT

Thus feature transformation is done for the given dataset.

















