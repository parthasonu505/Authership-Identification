import streamlit as st
from PIL import Image,ImageEnhance
st.title("Authorship Detection Demo")
st.text("Build with Streamlit and Natural Language Processiong")
activities = ["Upload","DEMO","About"]
choice = st.sidebar.selectbox("Select Activty",activities)
if choice =="Upload":
    dataset1= st.file_uploader("Upload Train Dataset",type=['csv'])
    dataset12= st.file_uploader("Upload Test Dataset",type=['csv'])
if choice == 'DEMO':
    st.subheader("Authorship Detection")
    c = st.sidebar.radio("Steps",["Dataset","Data Preprocessing",'Data Visualization',"Some Facts","Statistics","Accuracy"])
    if c=="Dataset":
        st.text(':::::::::First Five Row Of The DataSet Is Here:::::::::::::')
        f = open("C:\\Users\\user1\\Desktop\\Final year project\\train\\dataset_head.txt", "r")
        xx=f.read()
        st.text(xx)
    elif(c=="Data Preprocessing"):
        if (not st.button('See DataSet')):
            st.text(' ')
            st.text(' ')
            st.text(' ')
            st.subheader('Removing Punctuation Marks , Lemmatisation & Stop Word. . .')
        else:
            st.text(':::::::::First Five Row Of The DataSet Is Here:::::::::::::')
            f = open("C:\\Users\\user1\\Desktop\\Final year project\\train\\dataset_head21.txt", "r")
            xx=f.read()
            st.text(xx)
    elif(c=='Data Visualization'):
        rate = st.sidebar.slider("size",500,1000,600)
        task = ["BoxPlot","Unique Ratio Graph","Word Length Graph","Sentiment Analyser Graph","Word Cloud"]
        ch = st.sidebar.selectbox("Find Type",task)
        if(ch=='BoxPlot'):
            st.text("This is the Box Plot on how much word used by a Author")
            im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\boxplot1.png')
            st.image(im,width=rate)
        elif(ch=='Unique Ratio Graph'):
            st.text("This is the Unique Ratio Graph of each Author")
            im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\unique_ratio1.png')
            st.image(im,width=rate)
        elif(ch=='Word Length Graph'):
            st.text("This is the Word Length Graph of each Author")
            im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\average_word_length1.png')
            st.image(im,width=rate)
        elif(ch=='Sentiment Analyser Graph'):
            st.text("This is the Word Length Graph of each Author")
            im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\sentiment1.png')
            st.image(im,width=rate)
        elif(ch=='Word Cloud'):
            
            cp = st.sidebar.radio("Authors",["MWS","EAP",'HPL'])
            if(cp=='MWS'):
                st.text("Most frequent word by MWS")
                im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\wordcloud3.png')
                st.image(im,width=rate)
                
            elif(cp=='HPL'):
                st.text("Most frequent word by HPL")
                im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\wordcloud2.png')
                st.image(im,width=rate)
                
            elif(cp=='EAP'):
                
                st.text("Most frequent word by EAP")
                im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\wordcloud1.png')
                st.image(im,width=rate)
    elif(c=='Some Facts'):
        st.text('comming soon')
        age = st.slider('How old are you?', 0, 130, 25)
        st.write("I'm ", age, 'years old')
    elif(c=='Statistics'):
        task = ['Classification report','Confusion Matrix']
        ch = st.sidebar.selectbox("Find Type",task)
        if(ch=='Classification report'):
            st.text('::::::::::::::::The statistic About the model::::::::::::::::::')
            f = open("C:\\Users\\user1\\Desktop\\Final year project\\train\\classification_rept1.txt", "r")
            xx=f.read()
            st.text(xx)
        elif(ch=='Confusion Matrix'):
            st.text("::::::::::::::::Confusion Matrix About the Data model::::::::::::::::::")
            rate = st.sidebar.slider("size",500,1000,600)
            im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\confusion_matrix1.png')
            st.image(im,width=rate)
    elif(c=='Accuracy'):
        st.text('::::::::::::::::The statistic About the model::::::::::::::::::')
        f = open("C:\\Users\\user1\\Desktop\\Final year project\\train\\accuracy1.txt", "r")
        xx=f.read()
        st.subheader(xx)
if choice == 'About':
    st.subheader("About Author Detection App")
    st.markdown("Built with Streamlit by [Partha, Sucheta & Saikat]")
    im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\sucheta1.png')
    st.image(im,width=300)
    st.success("SUCHETA KAR ............................... Roll:12000117029")
    im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\saikat1.png')
    st.image(im,width=300)
    st.success("SAIKAT PAUL ............................... Roll:12000117055")
    im=Image.open('C:\\Users\\user1\\Desktop\\Final year project\\train\\partha1.png')
    st.image(im,width=300)
    st.success("PARTHA SARATHI PAL ........................ Roll:12000117074")
            
        
            
        
        
        
        
        
    