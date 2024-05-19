import pandas as pd
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

"""### **DATA EXPLORATION**"""

dataframe=pd.read_csv('/content/911.csv')

(dataframe.head())

print(dataframe.columns)

print(dataframe.shape)

print(dataframe.dtypes)

dataframe.isnull().sum()

print(dataframe.describe())

#sorting of timeStamp in ascending order
#dataframe['timeStamp'] = dataframe['timeStamp']

dataframe['timeStamp'] = pd.to_datetime(dataframe['timeStamp'], format='%Y-%m-%d %H:%M')

sorted_dataframe = dataframe.sort_values(by='timeStamp', ascending=False)

print(dataframe['timeStamp'])

# Sort the incident titles in ascending alphabetical order
sorted_dataframe = dataframe.sort_values(by='title', ascending=True)

dataframe['title'].head(25)

dataframe

dataframe.shape

"""### **DATA PREPROCESSING**

"""

dataframe.isna().sum()

df=dataframe.dropna()

df['timeStamp'] = df['timeStamp'].astype(str)

df[['Date','Time']]=df['timeStamp'].str.split(' ', expand=True)

df

df = df.drop(['e'],axis=1)

df.isna().sum()

df.dropna(inplace=True)
df.count()

df['desc'].str.split('Station', expand=True)[1].str.split(';', expand=True)[0]

#station from description
'''
df_station = pd.DataFrame()
df_station = df['desc'].str.split(';', expand=False)
print(df_station)

df_station = df_station.str[2]
df_station = df_station.str.extract(rf'Station\s+(.+)')
df_station
'''
df['Station_num'] = df['desc'].str.split('Station', expand=True)[1].str.split(';', expand=True)[0]

display(df)

df.isna().sum()

df['Station_num'].unique()

df['Call_Category'] = df.title.str.split(':', expand=True)[0]
df['Call_Reason'] = df.title.str.split(':', expand=True)[1].str.replace(' -', '')

df

dataframe_Ems = df
dataframe_Ems = dataframe_Ems[dataframe_Ems['Call_Category'] == 'EMS']
dataframe_Ems

'''
dataframe_nems = df
dataframe_nems = dataframe_nems[dataframe_nems['Station_num'].isnull() == True]
'''
dataframe_nems = df
dataframe_nems = dataframe_nems[dataframe_nems['Call_Category'] != 'EMS']


dataframe_nems

df

'''df['timeStamp']= pd.to_datetime(df['timeStamp'])
time = df['timeStamp'].iloc[0]
time.hour
df['Timing'] = df['timeStamp'].apply(lambda time: time.hour)'''

df['timeStamp']= pd.to_datetime(df['timeStamp'])
df['Timing'] = df['timeStamp'].dt.hour

dmap= {0: 'Night', 1:'Night', 2:'Night', 3:'Night', 4:'Night', 5:'Night', 6:'Morning', 7:'Morning', 8:'Morning', 9:'Morning', 10:'Morning', 11:'Morning', 12:'Afternoon', 13:'Afternoon', 14:'Afternoon', 15:'Afternoon', 16:'Afternoon', 17:'Evening', 18:'Evening', 19:'Evening', 20:'Evening', 21:'Night', 22:'Night', 23:'Night'}

df['Hour'] = df.timeStamp.dt.hour
df['Month'] = df.timeStamp.dt.month
df['DayOfWeek'] = df.timeStamp.dt.weekday

df['Timing'] = df['Timing'].map(dmap)

df

from google.colab import files
df.to_csv('911_cleaned.csv')
files.download('911_cleaned.csv')

"""#**DATA VISUALIZATION**

1. most called station
2. most emergency reasons of calls
3. count calls by reason
4. Day of Week column with the hue based off of the Reason column
5. same for month reason wise
6. line plot- count of calls
7. linear fit on the number of calls per month
8. heatmap

# **BAR PLOT: INDICATING NUMBER OF CALLS EACH MONTH**
"""

plt.figure(figsize=(15,9))

plt.hist(df.Month,bins=12,color="#b30000")
plt.title("Frequency Distribution per month")
plt.xlabel("Month")
plt.ylabel("Frequency")

"""# **MOST CALLED STATIONS - EMS**"""

most_called_ems = dataframe_Ems.Station_num.str.replace(':', '').value_counts()[:10]
plt.figure(figsize=(10, 6))
x = list(most_called_ems.index)
y = list(most_called_ems.values)
x.reverse()
y.reverse()

plt.title("Most Called Stations")
plt.ylabel("Station")
plt.xlabel("Number of calls")

plt.barh(x, y, color="#b30000")

"""# **What are the top 10 townships for 911 calls?**"""

dataframe.twp.value_counts().head(10)

"""# **MOST CALLS BY REASONS**"""

most_reasons_call = df.Call_Reason.value_counts()[:10]
plt.figure(figsize=(10,6))
x = list(most_reasons_call.index)
y = list(most_reasons_call.values)
x.reverse()
y.reverse()

plt.title("Most emergency reasons of calls")
plt.ylabel("Reasons")
plt.xlabel("Number of calls")

plt.barh(x, y, color="#ffbfbf")

"""# **BAR PLOT: INDICATING FREQUENCY OF CALLS PER EACH CATEGORY**"""

plt.figure(figsize = (10,8))
cs=["#b30000","#ffbfbf","#ff8080"]
listOfCategories = df['Call_Category'].unique()
listOfCountByCategory = []
for i in listOfCategories:
  listOfCountByCategory.append(df['Call_Category'].value_counts()[i])

plt.bar(listOfCategories,listOfCountByCategory, color=cs)

"""# **BAR PLOT: INDICATING FREQUENCY OF CALL BY TIME OF DAY**"""

plt.figure(figsize = (10,8))
cs=["#b30000","#ffbfbf","#ff8080"]
listOfHours = df['Timing'].unique()
listOfCountByHours = []
for i in listOfHours:
  listOfCountByHours.append(df['Timing'].value_counts()[i])

plt.bar(listOfHours,listOfCountByHours, color=cs)

"""# **NUMBER OF CALLS EACH DAY**"""

cs=["#b30000","#ffbfbf","#ff8080"]
plt.figure(figsize=(10, 8))
sns.countplot(x=df.DayOfWeek, hue=df.Call_Category,palette=cs)

byWeek = df.groupby('DayOfWeek').count()
byWeek

byWeek.twp.plot(figsize=(10, 8), color="#b30000")

"""# **REGRESSION MODEL**"""

plt.figure(figsize=(12, 8))
sns.lmplot(x='DayOfWeek',y='twp',data=byWeek.reset_index(), scatter_kws={'color':
"#b30000"})

"""# **NUMBER OF EMERGENCY CALLS EACH MONTH**"""

# Commented out IPython magic to ensure Python compatibility.
cs=["#b30000","#ffbfbf","#ff8080"]
plt.figure(figsize=(10, 8))
sns.countplot(x=df.Month, hue=df.Call_Category,palette=cs)

# %matplotlib inline
sns.set_style("whitegrid")

byMonth = df.groupby('Month').count()
byMonth

byMonth.twp.plot(figsize=(10, 8), color="#b30000")

"""# **HEAT MAP OF CALLS**"""

dayHour = df.groupby(by=['DayOfWeek','Hour']).count()['Call_Category'].unstack()
dayHour.head()

plt.figure(figsize=(12, 8))
sns.heatmap(dayHour)

