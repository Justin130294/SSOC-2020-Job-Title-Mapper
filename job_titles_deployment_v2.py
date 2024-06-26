
'''Import Libraries'''
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import os
import torch
import numpy as np
import gdown

# Remove tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Specify model directory
FINETUNED_MODEL_DIR = 'model'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
@st.cache_resource
def download_and_load_model():
    model_path = 'Job Mapper'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        # Google Drive URL
        url = "https://drive.google.com/drive/folders/1zYfaWYa2hTMvVqnv4FSojsUNcymj_Uca"
        output_folder = os.path.join(model_path, 'model.zip')
        gdown.download_folder(url)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

@st.cache_resource
def load_embeddings():
    return SentenceTransformer('BAAI/bge-base-en-v1.5')

# Create function to preprocess inputs
@st.cache_resource
def preprocess_function(job_title):
    PREFIX = 'correct job title: '
    SUFFIX = ''
    inputs = PREFIX + job_title.lower() + SUFFIX
    model_inputs = finetuned_tokenizer(inputs, max_length=128,
                                       padding='max_length', return_tensors='pt',
                                       truncation=True, return_attention_mask=False)
    return model_inputs

@st.cache_resource
def correct_title(job_title):
    inputs = preprocess_function(job_title)
    finetuned_model.to(device)
    pred = finetuned_model.generate(inputs.input_ids, max_new_tokens=128)
    output = finetuned_tokenizer.batch_decode(pred, skip_special_tokens=True)
    return output

@st.cache_resource
def generate_job_titles(job_title):
    corrected_title = correct_title(job_title)
    title_embeddings = embeddings.encode(corrected_title, convert_to_tensor=True, normalize_embeddings=True)
    alpha_index_embeddings = embeddings.encode(alpha_index_titles_list, convert_to_tensor=True, normalize_embeddings=True)
    job_suggestions_csim = util.pytorch_cos_sim(title_embeddings, alpha_index_embeddings)
    job_suggestions_csim = job_suggestions_csim.cpu().numpy()
    temp = list(map(lambda x: np.argsort(-x).tolist(), job_suggestions_csim))
    indices = temp[0]
    return indices, sorted(job_suggestions_csim)

def filter_job_titles(indices, job_suggestions_csim, threshold):
    temp = list(filter(lambda x: job_suggestions_csim[0][x] >= threshold, indices))
    occupations = []
    cosine_similarity = []
    for index in temp:
        cosine_similarity.append(job_suggestions_csim[0][index])
        occupations.append(alpha_index_titles_list[index])
    output_df = pd.DataFrame({"Occupations": occupations, "Cosine Similarity Score": cosine_similarity})
    output_df.index = np.arange(1, len(output_df) + 1)
    return output_df

@st.cache_data
def get_titles():
    alpha_index_data = pd.read_excel('alpha_index.xlsx')
    alpha_index_df = pd.DataFrame(alpha_index_data)
    return list(alpha_index_df['titles'])
alpha_index_titles_list = get_titles()

##### UI Implementation #####
# Load the model, tokenizer and embedding
with st.spinner('Loading model...'):
    finetuned_model,finetuned_tokenizer  = download_and_load_model()
    embeddings = load_embeddings()
    embeddings.max_seq_length = 512
    embeddings.tokenizer.padding_side = 'right'

# Add headers
st.header("SSOC 2020 Job Title Mapper")

# Enter text
job_title = st.text_input(label="Job Title", max_chars = 64,
              value = '',
              placeholder='Enter job title')

if job_title != '':
    # threshold = st.slider('Adjust Cosine Similarity Threshold', value=0.8, min_value=0.7, max_value=0.9, step=0.1)
    threshold = 0.6
    with st.spinner("Generating possible job titles"):
        indices, cos_sim_scores = generate_job_titles(job_title)
        output = filter_job_titles(indices, cos_sim_scores, threshold)
        st.write(output)
