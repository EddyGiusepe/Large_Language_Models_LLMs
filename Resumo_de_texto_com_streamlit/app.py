import streamlit as st
from transformers import pipeline

from transformers import T5Tokenizer
from transformers import T5Model, T5ForConditionalGeneration

model_name = 'phpaiola/ptt5-base-summ-xlsum'  #'phpaiola/ptt5-base-summ-xlsum'
tokenizer_name = 'unicamp-dl/ptt5-base-portuguese-vocab'

tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#@st.cache(persist=True)
def load_summarizer():
    model1 = pipeline(task="summarization",
                     model=model,
                     tokenizer=tokenizer,
                     framework='pt',
                     device=0)
    return model1


def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')
    
    sentences = inp_str.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks


summarizer = load_summarizer()
st.title("App que resume o Texto")
sentence = st.text_area('Por favor, cole o texto a resumir :', height=30)
button = st.button("Resumir")

max = st.sidebar.slider('Comprimento Máx. do resumo gerado', 50, 500, step=10, value=150)
min = st.sidebar.slider('Comprimento Mín.. do resumo gerado', 10, 450, step=10, value=50)
do_sample = st.sidebar.checkbox("Fazer Amostragem Estocástica", value=False)
# do_sample: False, o modelo gera o resumo mais provável. True, o modelo usa AMOSTRAGEM ESTOCÁSTICA para fazer o resumo
with st.spinner("Gerando resumo do Texto . . ."):
    if button and sentence:
        chunks = generate_chunks(sentence)
        res = summarizer(chunks,
                         max_length=max, 
                         min_length=min, 
                         do_sample=do_sample)
        text = ' '.join([summ['summary_text'] for summ in res])
        # st.write(result[0]['summary_text'])
        st.write(text)
