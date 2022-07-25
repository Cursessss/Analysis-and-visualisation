#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install pandas


# In[2]:


conda install matplotlib


# In[3]:


conda install pycountry


# In[4]:


conda install plotly


# In[5]:


conda install seaborn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import pandas as pd


# In[8]:


df = pd.read_csv('/home/curse/Downloads/aact_eaa01_page_linear.csv')


# In[9]:


df


# In[10]:


df.columns


# In[11]:


col = df.columns[6:-1]
col


# In[12]:


df = df[col]
df


# In[13]:


df.info()


# In[14]:


df.loc[:, 'geo'] = df['geo'].astype('category')
df.info()


# In[15]:


df['geo'].unique()


# In[16]:


df['geo'] = df['geo'].cat.add_categories(["GB", "GR"])


# In[17]:


pd.options.mode.chained_assignment = None  # swich of the warnings
mask = df['geo'] == 'UK' # Binary mask
df.loc[mask, 'geo'] = "GB" # Change the values for mask
df


# In[19]:


mask = df['geo'] == 'EL'
df.loc[mask, 'geo'] = 'GR'
df


# In[21]:


import pycountry


# In[22]:


list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]  # create a list of country codes
print("Country codes", list_alpha_2)

def country_flag(df):
    '''
    df: Series
    return: Full name of country or "Invalide code"
    '''
    if (df['geo'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['geo']).name
    else:
        print(df['geo'])
        return 'Invalid Code'

df['country_name']=df.apply(country_flag, axis = 1)
df


# In[23]:


mask = df['country_name'] != 'Invalid Code'
df = df[mask]
df


# In[24]:


df.info()


# In[25]:


df.describe()


# In[26]:


df.describe(include=['category'])


# In[27]:


df['country_name'].value_counts()


# In[28]:


pt_country = pd.pivot_table(df, values= 'OBS_VALUE', index= ['TIME_PERIOD'], columns=['country_name'], aggfunc='sum', margins=True)
pt_country


# In[29]:


pt_country.describe()


# In[30]:


pt = pd.pivot_table(df, values= 'OBS_VALUE', index= ['country_name'], columns=['TIME_PERIOD'], aggfunc='sum', margins=True)
pt


# In[31]:


pt.describe()


# In[32]:


pt.iloc[-1][:-1].plot()


# In[33]:


pt['All'][:-1].plot.bar(x='country_name', y='val', rot=90)


# In[34]:


pt.loc['Sweden'][:-1].plot()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(pt.columns)-1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots() # Create subplots
rects1 = ax.bar(x - width/2, pt.loc['Germany'][:-1], width, label='Germany') # parameters of bars
rects2 = ax.bar(x + width/2, pt.loc['France'][:-1], width, label='France')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('OBS_VALUE')
ax.set_xlabel('Years')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(pt.columns[:-1])
ax.legend()

fig.tight_layout()

plt.show()


# In[46]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(pt.columns)-1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots() # Create subplots
rects1 = ax.bar(x - width/2, pt.loc['Austria'][:-1], width, label='Austria') # parameters of bars
rects2 = ax.bar(x + width/2, pt.loc['Spain'][:-1], width, label='Spain')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('OBS_VALUE')
ax.set_xlabel('Years')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(pt.columns[:-1])
ax.legend()

fig.tight_layout()

plt.show()


# In[36]:


import seaborn as sns
d = pd.DataFrame(pt.loc['Sweden'][:-1])
print(d)
sns.regplot(x=d.index, y="Sweden", data=d,)


# In[37]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.reshape(d.index, (-1, 1)) # transform X values
y = np.reshape(d.values, (-1, 1)) # transform Y values
model.fit(X, y)


# In[38]:


X_pred= np.append(X, [2021, 2022, 2023])
X_pred = np.reshape(X_pred, (-1, 1))
# calculate trend
trend = model.predict(X_pred)

plt.plot(X_pred, trend, "-", X, y, ".")


# In[39]:


import plotly.express as px


# In[40]:


df


# In[41]:


import json
get_ipython().system('wget european-union-countries.geojson "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-agriculture-basic-statistical-analysis-and-geo-visualisation/european-union-countries.geojson"')
with open("european-union-countries.geojson", encoding="utf8") as json_file:
    EU_map = json.load(json_file)


# In[42]:


fig = px.choropleth(
    df,
    geojson=EU_map,
    locations='country_name',
    featureidkey='properties.name',    
    color= 'OBS_VALUE', 
    scope='europe',
    hover_name= 'country_name',
    hover_data= ['country_name', 'OBS_VALUE'],
    animation_frame= 'TIME_PERIOD', 
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
)


# In[43]:


fig.update_geos(showcountries=False, showcoastlines=False, showland=True, fitbounds=False)

fig.update_layout(
    title_text ="Agriculture Economic accounts",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)


# In[44]:


from IPython.display import HTML
HTML(fig.to_html())


# In[ ]:




