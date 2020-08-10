#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


train_data=pd.read_csv('C:/Users/HP/Documents/Ashu/Hackerearth/train.csv')


# In[4]:


train_data.head()


# In[5]:


test_data=pd.read_csv('C:/Users/HP/Documents/Ashu/Hackerearth/test.csv')


# In[6]:


test_data.head()


# In[7]:


train_data.describe()


# In[8]:


train_data.shape


# In[9]:


train_data.info()


# In[32]:


X=train_data.drop(columns=['Severity','Accident_ID','Max_Elevation'])
X.head()


# In[33]:


y=train_data.Severity


# In[34]:


from sklearn.model_selection import train_test_split


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[112]:


from sklearn.ensemble import RandomForestClassifier


# In[113]:


from sklearn.metrics import classification_report,confusion_matrix


# In[114]:


rfc=RandomForestClassifier(n_estimators=200)


# In[115]:


rfc.fit(X_train,y_train)


# In[116]:


rfc_pred=rfc.predict(X_test)


# In[117]:


rfc_pred


# In[118]:


print(confusion_matrix(y_test,rfc_pred))


# In[119]:


print(classification_report(y_test,rfc_pred))


# In[120]:


rfc.score(X_test,y_test)


# In[121]:


output=pd.DataFrame(rfc_pred,columns=['Severity'])
#output.to_csv('Submission.csv',index=False)


# In[122]:


output


# In[123]:



test_data1 = test_data.drop(columns=['Accident_ID','Max_Elevation'])


# In[124]:


test_data1


# In[125]:


pred_final=rfc.predict(test_data1)
pred_final


# In[126]:


Accident_ID = test_data['Accident_ID']
output = pd.DataFrame({'Accident_ID':Accident_ID,
                      'Severity':pred_final})


# In[127]:


output


# In[130]:


output.to_csv(r'C:\Users\HP\Desktop\Hacker\Result1.csv',index=False)

