import streamlit as st
import numpy as np 
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt 
import plotly
import plotly.figure_factory as ff 
import time


#Set title
st.title("Our First Streamlit App")
st.subheader('My Data Science App')
image=Image.open("data science.jpg")
st.image(image,use_column_width=True)

# #DF
st.info("Data Frame:")
df=pd.DataFrame(np.random.rand(10,20), columns=(f'col {i}' for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))
st.text("---"*100)

# #Line chart
st.info("Line chart:")
chart_data=pd.DataFrame(np.random.randn(20,3), columns=['a','b','c'])
st.line_chart(chart_data)
st.text("---"*100)

# #Area chart
st.info("Area chart:")
st.area_chart(chart_data)
st.text("---"*100)

# #Bar chart
st.info("Bar chart:")
chart_data=pd.DataFrame(np.random.randn(50,3), columns=['a','b','c'])
st.bar_chart(chart_data)
st.text("---"*100)

# #Matplotlib
import matplotlib.pyplot as plt 
st.info("Matplotlib histogram chart:")
arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.text("---"*100)

# #Plotly
import plotly
import plotly.figure_factory as ff 
st.info("Plotly distribution plot:")
x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)-2
hist_data=[x1,x2,x3]
group_labels=['Group1','Group2','Group3']
fig=ff.create_distplot(hist_data,group_labels,bin_size=[.2,.25,.5])
st.plotly_chart(fig,use_container_width=True)
st.text("---"*100)

# #Maps
st.info("Maps:")
df=pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4], columns=['lat','lon'])
st.map(df)
st.text("---"*100)

# # creating buttons 
st.info("UI:")
if st.button("Say hello"):
	st.write("hello is here")
else:
	st.write("why are you here")
st.text("---"*100)

# # raio button
genre=st.radio("What is your favourite genre?", ('Commedy','Drama','Documentary'))
if genre=='Commedy':
	st.write("Oh you like Commedy")
elif genre=='Drama':
	st.write("Yeah Drama is cool")
else:
	st.write(" i see!!")
st.text("---"*100)

# # Select button
option=st.selectbox("How was your night?",('Fantastic','Awesome','So-so'))
st.write("Your said your night was:",option)
st.text("---"*100)

# # multi select button 
option=st.multiselect("How was your night, you can select multiple choice?",('Fantastic','Awesome','So-so'))
st.write("Your said your night was:",option)
st.text("---"*100)

# # slider
age=st.slider('How old are you?',0,100,18)
st.write("Your age is : ",age)
st.text("---"*100)

# # range slider 
values=st.slider('Select a range of values',0, 200,(15,80))
st.write('You selected a range between:', values)
st.text("---"*100)

# # number input 
number=st.number_input('Input number')
st.write('The number you inputed is:',number)
st.text("---"*100)

# # color picker 
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)
st.text("---"*100)

# # side bar select box
add_sidebar=st.sidebar.selectbox("What is your favourite course?",('A course from TDS on building Data Web APP','Others', 'Am not sure'))

# # Progress Bar
st.info("Progress Bar:")
my_bar=st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)
st.text("---"*100)

# # spinner
st.info("Spinner:")
with st.spinner('wait for it...'):
	time.sleep(5)
st.success('successful')
st.text("---"*100)

st.balloons()

# #File uploader
st.info("File Uploader:")
upload_file=st.file_uploader("Choose a csv file", type='csv')
if upload_file is not None:
	data=pd.read_csv(upload_file)
	st.write(data)
	st.success("successfully uploaded")
else:
	st.markdown("Please  upload a CSV file")
st.text("---"*100)







