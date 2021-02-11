import streamlit as st
import pandas as pd
import plotly.express as px
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

with open('model.pickle', 'rb') as file:  
    modelo = pickle.load(file)


@st.cache
def get_data():
    url = 'https://raw.githubusercontent.com/MSR1805200/Portifolio/main/aprendizado_maquina_ufsc/projeto_final/data/apoio.xls'
    return pd.read_excel(url)

def plot_freq(title,x,y,color,data):
    st.plotly_chart( px.bar(x = data.value_counts().sort_values(ascending= False).index,
                            y = data.value_counts().sort_values(ascending= False).values,
                            title = title,
                            labels = {'x' : x,'y' : y}
                            )
                     )
def plot_map(title,x,y,color,data):
    
    st.plotly_chart( px.choropleth(
                            locations = data.value_counts().sort_values(ascending = False).index,
                            color = data.value_counts().sort_values(ascending = False).values,
                            color_continuous_scale = color,
                            labels = {'color' : y,'locations' : x},
                            title = title)
                         )

df = get_data()

vector = CountVectorizer(strip_accents = 'unicode',lowercase= True,stop_words= 'english')

vector.fit(df['OriginalTweet'].values)




st.title('Projeto ponta a ponta : Classificando sentimentos relacionados ao covid')

image = Image.open('raiva.jpg')

st.image(image, caption='Fonte: <https://mumbrella.com.au/wp-content/uploads/2016/11/angry-social-media-reputation-management.jpg>',
              width=None, use_column_width=None, channels='RGB')

st.markdown('Projeto de mineração de dados')
st.write('Aqui está um breve descrição sobre o projeto final do curso de Aprendizado de Máquinas da Escola de Verão da UFSC, ministrada pelo professor Edson: '
             'ele surge no contexto da pandemia do novo coronavirus COVID-19, onde uma das maiores redes sociais, o Twitter, '
             'foi um dos grandes meios responsável pela dissiminação de opiniões sobre a pandemia. '
             'O presente projeto treinou um modelo para identificar se um tweet qualquer expressa um sentimento positivo (0), negativo (1) ou neutro (2) quanto ao tema.' )

st.sidebar.title('Menu')

pag = st.sidebar.selectbox('Selecione',['Análise','Predição'])

if pag =='Análise':
    colunas = st.multiselect('Selecione as colunas', df.columns.tolist())
    if len(colunas) == 0 :
        colunas = df.columns.tolist()
    
    st.dataframe(df[colunas])
    link = '[Acessar fonte do dataset](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification)'
    st.markdown(link, unsafe_allow_html=True)

    fig = st.selectbox('Selecione o gráfico',['Sentiment Frequency','Month Frequency',
                                              'Month Frequency with Positive sentiment',
                                              'Month Frequency with Negative sentiment',
                                              'Month Frequency with Neutral sentiment'
                                              ])
    
    if fig == 'Sentiment Frequency' :
        
        plot_freq('Frequência dos sentimentos','Sentimentos','Quantidade','blues',df['Sentiment'])

    elif fig == 'Month Frequency':

        plot_freq('Frequência dos Meses','Meses','Quantidade','blues',df['Month'])
        
    elif fig == 'Month Frequency with Positive sentiment':
        
        positive_freq = df[df['Sentiment'] == 0]

        plot_freq('Frequência dos Meses com o sentimento positivo','Meses','Quantidade','Greens',positive_freq['Month'])
        
    elif fig == 'Month Frequency with Negative sentiment':
        negative_freq = df[df['Sentiment'] == 1]

        plot_freq('Frequência dos Meses com o sentimento negativo','Meses','Quantidade','Reds',negative_freq['Month'])
        
    elif fig == 'Month Frequency with Neutral sentiment':
         neutral_freq = df[df['Sentiment'] == 2]
        
         plot_freq('Frequência dos Meses com o sentimento Neutro','Meses','Quantidade','ylorbr',neutral_freq['Month'])
        
    fig2 = st.selectbox('Selecione o mapa',['Frequency','Positive Frequency','Negative Frequency','Neutral Frequency'])

    if fig2 == 'Frequency':
        nao_nulos = df['Country'][df['Country'].notnull()]

        plot_map('Frequência dos tweets agrupados pelos países','Local','Quantidade','blues',nao_nulos)
    
    if fig2 == 'Positive Frequency':

        nao_nulos_pos = df['Country'][df['Country'].notnull()][df['Sentiment'] == 0]

        plot_map('Frequência dos tweets positivos agrupados pelos países','Local','Quantidade','Greens',nao_nulos_pos)
        
    elif fig2 == 'Negative Frequency':
        
         nao_nulos_neg = df['Country'][df['Country'].notnull()][df['Sentiment'] == 1]

         plot_map('Frequência dos tweets negativos agrupados pelos países','Local','Quantidade','Reds',nao_nulos_neg)
         
    elif fig2 == 'Neutral Frequency':

         nao_nulos_neu = df['Country'][df['Country'].notnull()][df['Sentiment'] == 2]

         plot_map('Frequência dos tweets neutros agrupados pelos países','Local','Quantidade', 'ylorbr',nao_nulos_neu)

elif pag == 'Predição':
    
    st.header('Predição')
        
    frase = st.text_input('Digite a frase em inglês')

    frase_process =vector.transform([frase])

    pred = modelo.predict(frase_process)
    
    if pred[0] == 0:
        st.success("Positivo")
    elif pred[0] == 1:
        st.error("Negativo")
    else:
        st.warning("Neutro")
    
    




