import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
matplotlib.use('Agg')

from PIL import Image

#Set title

st.title('My Data Science App')
image = Image.open('data science.jpg')
st.image(image,use_column_width=True)



def main():
	activities=['EDA','Visualisation','model','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	


#DEALING WITH THE EDA PART


	if option=='EDA':
		st.subheader("Exploratory Data Analysis")

		x,y,feature_names = get_data_set()
		if x is not None and y is not None:
			df = pd.DataFrame(x,columns=feature_names)
			st.write(df.head(50))

			if st.checkbox("Display shape"):
				st.write(df.shape)
			if st.checkbox("Display columns"):
				st.write(df.columns)
			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect('Select preferred columns:',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox("Display summary"):
				try:
					st.write(df1.describe().T)
				except UnboundLocalError:
					st.write(df.describe().T)
			if st.checkbox('Display Null Values'):
				st.write(df.isnull().sum())
			if st.checkbox('Display Correlation of data variuos columns'):
				st.write(df.corr())




#DEALING WITH THE VISUALISATION PART


	elif option=='Visualisation':
		st.subheader("Data Visualisation")

		x,y,feature_names = get_data_set()
		if x is not None and y is not None:
			df = pd.DataFrame(x,columns=feature_names)
			st.write(df.head(50))
			st.warning("Select columns for all the data sets excluding iris before visualizing, Deselect viz before selecting the columns")
			if st.checkbox('Select Multiple columns to plot'):
				selected_columns=st.multiselect('Select your preferred columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox('Display Heatmap'):
				try:
					st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
					st.set_option('deprecation.showPyplotGlobalUse', False)
					st.pyplot()
				except UnboundLocalError:
					st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
					st.set_option('deprecation.showPyplotGlobalUse', False)
					st.pyplot()	
			if st.checkbox('Display Pairplot'):
				try:
					st.write(sns.pairplot(df1,diag_kind='kde'))
					st.set_option('deprecation.showPyplotGlobalUse', False)
					st.pyplot()
				except UnboundLocalError:
					st.write(sns.pairplot(df,diag_kind='kde'))
					st.set_option('deprecation.showPyplotGlobalUse', False)
					st.pyplot()
			


	# DEALING WITH THE MODEL BUILDING PART

	elif option=='model':
		st.subheader("Model Building")
		X,y,feature_names = get_data_set()
		if X is not None and y is not None:
			df = pd.DataFrame(X,columns=feature_names)
			st.write(df.head(50))
			seed=st.sidebar.slider('Seed',1,200)
			classifier_name=st.sidebar.selectbox('Select your preferred classifier:',('KNN','SVM','LR','naive_bayes','decision tree'))
			#calling the function
			params=add_parameter(classifier_name)
			#defing a function for our classifier
			clf=get_classifier(classifier_name,params)
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)
			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)
			st.write('Predictions:',y_pred)
			accuracy=accuracy_score(y_test,y_pred)
			st.write('Nmae of classifier:',classifier_name)
			st.write('Accuracy',accuracy)

#DELING WITH THE ABOUT US PAGE
	elif option=='About us':

		st.markdown('This is an interactive web page for our ML project, feel feel free to use it. This dataset is fetched from the UCI Machine learning repository. The analysis in here is to demonstrate how we can present our wok to our stakeholders in an interractive way by building a web app for our machine learning algorithms using different dataset.'
			)


		st.balloons()
	# 	..............


def get_data_set():
	name=st.sidebar.selectbox('Select dataset',('Iris','Wine','Boston','Breast Cancer'))
	if name=='Iris':
		data=datasets.load_iris()
	elif name=='Wine':
		data=datasets.load_wine()
	elif name=='Boston':
		data=datasets.load_boston()
	elif name=='Breast Cancer':
		data=datasets.load_breast_cancer()
	else:
		pass
	x=data.data
	y=data.target
	feature_names = data.feature_names
	return x,y,feature_names

def add_parameter(name_of_clf):
	params=dict()
	if name_of_clf=='SVM':
		C=st.sidebar.slider('C',0.01, 15.0)
		params['C']=C
	elif name_of_clf=='KNN':
		K=st.sidebar.slider('K',1,15)
		params['K']=K
	return params

def get_classifier(name_of_clf,params):
	clf= None
	if name_of_clf=='SVM':
		clf=SVC(C=params['C'])
	elif name_of_clf=='KNN':
		clf=KNeighborsClassifier(n_neighbors=params['K'])
	elif name_of_clf=='LR':
		clf=LogisticRegression()
	elif name_of_clf=='naive_bayes':
		clf=GaussianNB()
	elif name_of_clf=='decision tree':
		clf=DecisionTreeClassifier()
	else:
		st.warning('Select your choice of algorithm')
	return clf


if __name__ == '__main__':
	main() 



