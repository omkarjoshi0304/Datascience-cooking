import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer



rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)

d_c=['greek','southern_us','filipino','indian','jamaican','spanish','italian','mexican','chinese','british','thai','vietnamese','cajun_creole','brazilian','french','japanese','irish','korean','moroccan','russian','greek']

df=pd.read_json("C:/Users/admin/Downloads/whats-cooking/train.json/train.json")

#print(df)

print(df["cuisine"].unique())
x=df["ingredients"]
y=df["cuisine"].apply(d_c.index)

df['all_ingredients']=df['ingredients'].map(";".join)
cv=CountVectorizer()
X=cv.fit_transform(df["all_ingredients"].values)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


rf.fit(X_train,y_train)
r_pred=rf.predict(X_test)


print("random forest:",accuracy_score(y_test,r_pred))

#output
'''C:\Users\admin\PycharmProjects\pythonProject\venv\Scripts\python.exe C:/Users/admin/PycharmProjects/pythonProject/data7.py
['greek' 'southern_us' 'filipino' 'indian' 'jamaican' 'spanish' 'italian'
 'mexican' 'chinese' 'british' 'thai' 'vietnamese' 'cajun_creole'
 'brazilian' 'french' 'japanese' 'irish' 'korean' 'moroccan' 'russian']
random forest: 0.7583909490886235

Process finished with exit code 0'''