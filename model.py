from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.vectorstores import Chroma
from google.cloud import storage
from google.cloud.storage import Client, transfer_manager

from langchain.chains import RetrievalQA
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.llms import AzureOpenAI
import os
from pathlib import Path
import re,ast
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = "874f326ebe0e41c2906e435b3c60cd4c"
os.environ["AZURE_OPENAI_ENDPOINT"]="https://cda2rai-openi-resume-filter.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
deployment_name="gpt-35-turbo"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "gcp_service_act.json"

llm = AzureChatOpenAI(azure_deployment=deployment_name, 
             temperature=0, 
             max_tokens = 500)

summary  = ''
def handle_summarization(fileName): 
    global summary
    if fileName.lower().endswith('.pdf'):
        loader=PyPDFLoader("./testapidata/"+fileName)
    elif fileName.lower().endswith('.txt'):
        loader=TextLoader("./testapidata/"+fileName)
    elif fileName.lower().endswith('.docx'):
        loader=Docx2txtLoader("./testapidata/"+fileName)
    pages =loader.load()
    prompt_template = '''Summary Should include all skills, certifications, projects and past experience for the following {text}   
    and it should be betwen 100 to 120 words and it should include total number of years of work experience they has and give it as separate key
     and value, key is noOfExp and value is 4
    '''
    prompt = PromptTemplate(template=prompt_template,
                        input_variables=["text"])

    chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=prompt)
    summary = chain.run(pages)
    return summary

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def handle_experience(summary):
    pattern = r'(\d+)\s*(year|years)'

# Search for the pattern in the text
    match = re.search(pattern, summary)

    # If a match is found, extract the number of years
    if match:
        years = int(match.group(1))
        return years
    else:
        return "Undefined"

def handle_skills(summary):
    docs = get_text_chunks_langchain(summary)
    persist_directory = 'db'
    db = Chroma.from_documents(documents=docs, embedding=AzureOpenAIEmbeddings(), persist_directory=persist_directory)
    query='''You an intelligent machine who can parse the given pdf and can you answer me my questions in this format skills : {skills of person}
     output should be object where key is skills and value as a list of all skills'''

    chain=RetrievalQA.from_chain_type(llm=AzureOpenAI(deployment_name=deployment_name),retriever=db.as_retriever(),
                                 chain_type="stuff")
    score = chain.run(query)
    dictionary_texts = re.findall(r'\{(?:[^{}]*"skills": \[[^\]]*\][^{}]*)\}', score)

    # Extract skills from each dictionary
    skills = []
    for dictionary_text in dictionary_texts:
        dictionary = eval(dictionary_text)  # Convert text to dictionary
        if 'skills' in dictionary:
            skills.extend(dictionary['skills'])

    # Deduplicate skills
    unique_skills = list(set(skills))
    return unique_skills

df=pd.DataFrame(columns=["Resume Name","Profile Summary","Experience","Skills"])

directory = Path('./testapidata')
count=0

def get_files_from_google(bucket_name,destination_directory="./testapidata",workers=8, max_results=1000):
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=max_results)]

    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either None or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))


def get_bucket_name(path):
    pattern = r'https://console\.cloud\.google\.com/storage/browser/([^/]+)'
    match = re.match(pattern, path)
    if match:
        bucket_name = match.group(1)
        return bucket_name
    else:
        return None

def get_summary_exp_skills (path):

    df=pd.DataFrame(columns=["Resume Name","Profile Summary","Experience","Skills"])
    bucket_name = get_bucket_name(path)
    get_files_from_google(bucket_name)

    for item in directory.iterdir():
        # Check if the item is a file
        if item.is_file():
            row={"Resume Name":"","Profile Summary":"","Experience":"","Skills":""}
            row["Resume Name"]=item.name
            try:
                row["Profile Summary"]=handle_summarization(item.name)
            except:
                row["Profile Summary"]='Not Found'
            try:
                row["Experience"]=handle_experience(summary)
            except:
                row["Experience"]="Not Found"
            try:
                row["Skills"]=handle_skills(summary)
            except:
                row["Skills"]="Not Found"
            temp_df=pd.DataFrame([list(row.values())],columns=list(row.keys()))
            df=pd.concat([df,temp_df],ignore_index=True)
            df.reset_index()

    file_path = "data.csv"
    df.to_csv(file_path, index=False)

def get_jobdescription(prompt):

    conversation_buf = ConversationChain(
    llm=llm)
    prompt_template = PromptTemplate.from_template(
    template='''Assume You are Job description genarator, your job is to genarate Job descripton based on user promps
    job description should include skills, qulaifications, experience(if given by user in prompt), roles and responsibilities
    so the prompt is {prompt} please genarate job description based on the prompt i need in  object format where keys are skills 
    qualifications, experience, then roles and responsibilities, values should be associated with each key as a list
    ''')
    prompt = prompt_template.format(prompt=prompt)
    ai_questions=conversation_buf.run(prompt)
    score=ai_questions
    score=score.replace("'","")
    pattern = r'{[\s\S]*}'
    match = re.search(pattern, score)
    if match:
        report_dict = match.group()
        score=ast.literal_eval(report_dict)
        return(report_dict)
    else:
        return 'info:Report Not Fount Pls Try again'

def get_resumes(description,threshold,noOfMatches):
    jd=description
    df=pd.read_csv("data.csv")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1= model.encode(jd, convert_to_tensor=True)
    cos_ll=[]
    for i in range(len(df["Profile Summary"])):
        summ=df["Profile Summary"][i]
        #Compute embedding for both lists
        embedding_2 = model.encode(summ, convert_to_tensor=True)

        cos_sim = torch.sum(embedding_1 * embedding_2, dim=-1)

        simm=util.pytorch_cos_sim(embedding_1, embedding_2)[0][0].item()
        cos_ll.append(simm)
    df["Similarty Score"]=cos_ll
    df["Similarty Score"]=df["Similarty Score"]+(df["Similarty Score"]*30/100)
    sorted_df_desc = df.sort_values(by='Similarty Score', ascending=False)
    resultant_df = sorted_df_desc.iloc[0:noOfMatches]
    resultant_df.reset_index(inplace=True)
    confidence_score = resultant_df['Similarty Score'].mean()
    response_dict = dict()
    results=[]
    count=0
    for i in range(noOfMatches):
        count=count+1
        resumedict=dict()
        resumedict['id'] = count
        resumedict['path'] = resultant_df.loc[i]["Resume Name"]
        resumedict['score'] = resultant_df.loc[i]['Similarty Score']
        results.append(resumedict)
    response_dict['count'] = count
    response_dict['metadata'] = {"confidenceScore":confidence_score}
    response_dict['results'] = results
    return response_dict
    


    


    
