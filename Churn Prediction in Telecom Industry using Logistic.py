#!/usr/bin/env python
# coding: utf-8

# ## Churn Prediction in Telecommunication using Logistic Regression and Logit Boost
# 
# =======================================================================================================================

# ## Objective
# ============================================================
# ## Loading libraries and data
# ## Undertanding the data
# ## Visualize missing values
# ## Data Manipulation
# ## Data Visualization
# ## Data Preprocessing
# ## Standardizing numeric attributes
# ## Machine Learning Model Evaluations and Predictions
# ## KNN
# ## SVC
# ## Random Forest
# ## Logistic Regression
# ## Decision Tree Classifier
# ## AdaBoost Classifier
# =====================================================================================

# # Loading libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import missingno as ms
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[4]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:



from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# ## Loading data

# In[6]:


df=pd.read_csv(r"C:\Users\smkon\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ## Understanding the data

# In[7]:


df


# In[8]:


df.shape


# In[9]:


df.head(100)


# In[10]:


df.tail(100)


# In[11]:


df["gender"]


# In[12]:


df.columns.values


# In[13]:


df.dtypes


# In[14]:


df.describe


#  ## Visualize missing value

# In[15]:


##  Visualize missing values as a matrix 
ms.matrix(df)


# ## From the above visualisation we can observe that it has no peculiar pattern that stands out. In fact there is no missing data.

# In[16]:


df.isna().sum()


# ## Data manipulation 

# In[17]:


df1=df.drop(['customerID'],axis=1)


# In[18]:


df1


# ## On deep analysis, we can find some indirect missingness in our data (which can be in form of blankspaces). Let's see that!

# In[19]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# ## Here we see that the TotalCharges has 11 missing values. Let's check this data.

# In[20]:


df[df["tenure"]==0].index


# ## Let's delete the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

# In[21]:


df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index


# ## To solve the problem of missing values in TotalCharges column, I decided to fill it with the mean of TotalCharges values.

# In[22]:


df.fillna(df["TotalCharges"].mean())


# In[24]:


df.isnull().sum()


# In[82]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df.head()


# In[83]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()


# #### Data Visualization 

# In[25]:


g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
fig.show()


# ## * 26.6 % of customers switched to another firm.
#  ## * Customers are 49.5 % female and 50.5 % male.

# In[ ]:


df["Churn"][df["Churn"]=="No"].groupby(by=df["gender"]).count()


# In[26]:


plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# ## * There is negligible difference in customer percentage/ count who chnaged the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.

# In[ ]:


fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## * About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract

# In[27]:


labels = df['PaymentMethod'].unique()
values = df['PaymentMethod'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>Payment Method Distribution</b>")
fig.show()


# In[28]:


fig = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *Major customers who moved out were having Electronic Check as Payment Method.
# # * Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.

# In[29]:


df["InternetService"].unique()


# In[30]:


df[df["gender"]=="Male"][["InternetService", "Churn"]].value_counts()


# In[31]:


df[df["gender"]=="Female"][["InternetService", "Churn"]].value_counts()


# In[32]:


fig = go.Figure()

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [965, 992, 219, 240],
  name = 'DSL',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [889, 910, 664, 633],
  name = 'Fiber optic',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [690, 717, 56, 57],
  name = 'No Internet',
))

fig.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")

fig.show()


# # * A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
# ## * Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.

# In[33]:


color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>Dependents distribution</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *Customers without dependents are more likely to churn

# In[ ]:


color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="Partner", barmode="group", title="<b>Chrun distribution w.r.t. Partners</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *Customers that doesn't have partners are more likely to churn

# In[34]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>Chrun distribution w.r.t. Senior Citizen</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *It can be observed that the fraction of senior citizen is very less.
# ## *Most of the senior citizens churn.

# In[35]:


color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group", title="<b>Churn w.r.t Online Security</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *Most customers churn in the absence of online security,

# In[36]:


color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="PaperlessBilling",  title="<b>Chrun distribution w.r.t. Paperless Billing</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ### *Customers with Paperless Billing are most likely to churn.

# In[37]:


fig = px.histogram(df, x="Churn", color="TechSupport",barmode="group",  title="<b>Chrun distribution w.r.t. TechSupport</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ## *Customers with no TechSupport are most likely to migrate to another service provider.

# In[38]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="PhoneService", title="<b>Chrun distribution w.r.t. Phone Service</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# ### *Very small fraction of customers don't have a phone service and out of that, 1/3rd Customers are more likely to churn.

# In[40]:


fig = px.box(df, x='Churn', y = 'tenure')

# Update yaxis properties
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Tenure vs Churn</b>',
)

fig.show()


# ## *New customers are more likely to churn

# In[39]:


plt.figure(figsize=(25, 10))

corr = df.apply(lambda x: pd.factorize(x)[0]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)


# ## DATA PREPROCESING 

# In[41]:


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# In[42]:


df = df.apply(lambda x: object_to_int(x))
df.head()


# In[43]:


plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)


# In[44]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[45]:


X


# In[46]:


y


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)


# In[48]:


X_train


# In[49]:


X_test


# In[50]:


y_train


# In[51]:


y_test


# In[52]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[53]:


X


# In[54]:


y


# In[55]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)


# In[56]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)


# ### * Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range

# ## Standardizing numeric attributes

# In[58]:


df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std, color='c')


# In[59]:



## # Divide the columns into 3 categories, one ofor standardisation, one for label encoding and one for one hot encoding
cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # those that need one-hot encoding
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #those that need label encoding


# In[60]:


cat_cols_ohe


# In[61]:


cat_cols_le


# In[62]:


scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[63]:


X_train[num_cols] 


# In[64]:


X_test[num_cols]


# ##  Machine Learning Model Evaluations and Predictions

# ## K-Nearest Neighbor(KNN)

# In[65]:


knn_model = KNeighborsClassifier(n_neighbors = 11) 
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN accuracy:",accuracy_knn)


# In[66]:


print(classification_report(y_test, predicted_y))


# ## Support Vector Machine

# In[67]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# In[68]:


print(classification_report(y_test, predict_y))


# ## Random Forest

# In[69]:


model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =0, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[70]:


print(classification_report(y_test, prediction_test))


# In[71]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[72]:


y_rfpred_prob = model_rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='Random Forest',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();


# ### Logistic Regression

# In[73]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# In[74]:


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)


# In[75]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[76]:


y_pred_prob = lr_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show();


# ## Decision Tree Classifier

# In[77]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# ## *Decision tree gives very low score.

# In[78]:


print(classification_report(y_test, predictdt_y))


# ### AdaBoost Classifier

# In[79]:


a_model = AdaBoostClassifier()
a_model.fit(X_train,y_train)
a_preds = a_model.predict(X_test)
print("AdaBoost Classifier accuracy")
metrics.accuracy_score(y_test, a_preds)


# In[80]:


print(classification_report(y_test, a_preds))


# In[81]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, a_preds),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("AdaBoost Classifier Confusion Matrix",fontsize=14)
plt.show()


# In[ ]:





# In[ ]:




