from tasks import *
import pandas as pd
import streamlit as st
from collections import Counter

st.title('NLP Assist')
file_up = st.file_uploader("Upload a file", type='csv')
# file_up = 'assets/Precily_Text_Similarity.csv'
if file_up is not None:
    st.success("File uploaded successfully")
    df = pd.read_csv(file_up, encoding= 'unicode_escape')
    df.dropna(inplace=True)
    obj = []
    for i in df.columns:
        type = df[i].dtypes
        if type == 'object':
            obj.append(i)
        else:pass
    with st.sidebar:
        # Data Selection
        st.header("Variables")
        var_select = st.multiselect('Select Variables', obj,
                                  help='Select variables to perform further analysis')
        st.write(len(var_select), 'variables selected')
        
        # Exploratory Data Analysis
        st.subheader('Exploratory Data Analysis')
        eda_tasks = ['N Words', 'Language Detection', 
                    'Subjectivity', 'Polarity', 
                    'Sentiment']
        eda_select = st.multiselect('Select EDA Tasks', eda_tasks,
                    help='EDA Tasks can be performed only on selected variables')
        
        # Text Preprocessing
        st.subheader('Text Preprocessing')
        tp_tasks = ['Text cleaning', 'Stop Words removal','Tokenization','Stemming', 'Lemmatization']
        tp_select = st.multiselect('Select Text Preprocessing Tasks', tp_tasks,
                    help='Text Preprocessing Tasks can be performed only on selected variables and Text preprocessing is nothing but cleaning the text data to improve the quality of data for further analysis.')
       
        # NLP models
        st.subheader('NLP Models')
        models = ["Choose an option",'Named Entity Recognition', 'Text summarization', 'Keyword Extraction']
        model = st.selectbox('Select a model', models, 
                help='Select a model to perform further analysis')    

    # EDA
    for i in eda_select:
        if i == 'N Words':
            for i in var_select:
                df['n_words'+' '+str(i)] = df[i].apply(lambda x: nWords(str(x)))
        elif i == 'Language Detection':
            for i in var_select:
                df['languages'+' '+str(i)] = df[i].apply(lambda x: getLanguages(str(x)))
        elif i == 'Subjectivity':
            for i in var_select:
                df['subjectivity'+' '+str(i)] = df[i].apply(lambda x: getSubjectivity(str(x)))
        elif i == 'Polarity':
            for i in var_select:
                df['polarity'+' '+str(i)] = df[i].apply(lambda x: getPolarity(str(x)))
        elif i == 'Sentiment':
            for i in var_select:
                polarity = df[i].apply(lambda x: getPolarity(str(x)))
                df['sentiment'+' '+str(i)] = polarity.apply(getSentiment)
                counter_result = Counter(df['sentiment'+' '+str(i)])
                for key, value in counter_result.items():
                    st.write(str(i), key, 'Sentiment➡️', value)

   # Text Preprocessing                 
    for i in tp_select:
        if i == 'Text cleaning':
            for i in var_select:
                df['cleanedText'+' '+str(i)] = df[i].apply(textClean)
        elif i == 'Stop Words removal':
            for i in var_select:
                df['noStopWordsText'+' '+str(i)] = df[i].apply(stopWords)
        elif i == 'Tokenization':
            for i in var_select:
                df['tokenizedText'+' '+str(i)] = df[i].apply(tokenize)
        elif i == 'Stemming':
            for i in var_select:
                df['stemmedText'+' '+str(i)] = df[i].apply(stemming)
        elif i == 'Lemmatization':
            for i in var_select:
                df['lemmatizedText'+' '+str(i)] = df[i].apply(lemming)

    # Models                   
    if model == 'Named Entity Recognition':
        for i in var_select:
            df['NER'+' '+str(i)] = df[i].apply(ner)
    elif model == 'Text summarization':
        for i in var_select:
            df['Text Summarization'+' '+str(i)] = df[i].apply(transformer_summarizer)  
    elif model == 'Keyword Extraction':
        for i in var_select:
            df['Keyword Extraction'+' '+str(i)] = df[i].apply(lambda x: keywordExtractor(x)).astype(str)
    else:
        pass
    
    # Display Dataframe        
    button = st.radio('', ('👀 View Complete Dataframe',
                            '👀 View Specific-columed Dataframe'), 
                                horizontal=True,
                                label_visibility='collapsed')
    if button == '👀 View Complete Dataframe':
        st.dataframe(df)
        csv = df.to_csv(index=False)
        parquet = df.to_parquet(index=False)
    else:
        var_select_col = st.multiselect('Select Variables', df.columns)
        st.dataframe(df[var_select_col])
        csv = df[var_select_col].to_csv(index=False)
        parquet = df[var_select_col].to_parquet(index=False)

    # Download Dataframe
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='NLP_Assisted.csv',
        mime='text/csv',
        help='''Download the dataframe in CSV format. CSV file consumes more disk space and time for both read and write compared to parquet format '''
    )
    st.download_button(
        label="Download Parquet",
        data=parquet,
        file_name='NLP_Assisted.parquet',
        mime='text/csv',
        help='''Download the dataframe in parquet format. Parquet file consumes less disk space and time for both read and write '''
        )

    