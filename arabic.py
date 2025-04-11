import fitz  # import for the text extraction for pdf
from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from docx import Document

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_text():

    #testing it on actual docs
    '''full_text = ""
    for i in range(1,46): 
        document_name = "/Users/mariam/Desktop/nis2p/word/" + str(i) + ".docx"
        document = Document(document_name)
        for paragraph in document.paragraphs:
            full_text += paragraph.text + "\n"  
    return full_text.strip()'''
    
    
    #testing it on practice doc
    full_text = ""

    document = Document("try.docx")
    for paragraph in document.paragraphs:
        full_text += paragraph.text + "\n"  
    return full_text.strip()

def chunk_text(text):
    arabic_punct = ['.', '؟', '!', '،', '؛', ':']
    chunks = []
    the_string = ""
    letter_count = 0
    hit_limit = False

    for line in text.splitlines():
        for letter in line:
            the_string += letter
            letter_count += 1
            if letter_count >= 500:
                hit_limit = True  
            if hit_limit and letter in arabic_punct: # keep on going until I hit a punctuation mark
                chunks.append(the_string.strip())
                the_string = ""
                letter_count = 0
                hit_limit = False

    if the_string.strip():
        chunks.append(the_string.strip())

    return chunks
     
'''def generate_embeddings(text_chunks):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    print("hello")
    for chunk in text_chunks: #going through each chunk in the list
                        #tokenizer --> converts it into numbers; passed in the chunk, return_tensors keeps the right format
                                    # truncation incase it exceeeds 512 words, paddding to keep length the same, max_length
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad(): #not going to train the model
                    # now we feed the tokinized/numbered text into the model
                    #.last_hidden_state --> matrix full of vectors
                    #.mean(dim = 1) --> takes avg of vectors in 1 sentence (question will this reduce the meaning if you're taking the avg of a long sentence)
                    #.squeeze().numpy returns it back in a list for us
            outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(outputs) #each chunk now has its own embedding/list of nums
    return embeddings'''

def generate_embeddings(text_chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = []
    print("hello")
    for chunk in text_chunks: #going through each chunk in the list
        emb = list(model.encode(chunk))
        embeddings.append(emb)
    return embeddings


#Step 0: embeddings
folder_path = '/Users/mariam/Desktop/nis2p/word/converted_pdfs'
file_path = os.path.join(folder_path, 'practice.pdf')  
full_text = get_text()
text_chunks = chunk_text(full_text)
embeddings = generate_embeddings(text_chunks)
#writing to parquet
df = pd.DataFrame({'text': text_chunks, 'embedding': embeddings})
df.to_parquet('output_embeddings.parquet')

#Step 1/2: User input and vectorize it
user_query = input("Please enter a question: ")
query_chunks = chunk_text(user_query)  # just incase there is a mega long question
query_embedding = generate_embeddings(query_chunks)
query_embedding = np.array(query_embedding)
query_embedding = query_embedding.squeeze()
#Step 3: loop through parquet vectors computer dp
df = pd.read_parquet('output_embeddings.parquet')
dot_products = []
for index, row in df.iterrows():
    row_vector = np.array(row['embedding'])
    dp = np.dot(row_vector, query_embedding)
    dot_products.append((index, dp))


max_dp = dot_products[0][1]
max_index = dot_products[0][0] 

for i in range(1, len(dot_products)):
    cur = dot_products[i][1]
    if cur > max_dp:
        max_dp = cur
        max_index = dot_products[i][0] 

print("max dp: ", max_dp)
print("all of them to error check")
for item in dot_products:
    print(f"index: {item[0]}, dp: {item[1]}")

max_text = df.loc[max_index, 'text'] 
print("\nAnswer: ", max_text)

'''
saving code: 

def chunk_text(text):
    chunks = []
    lines = text.splitlines() #returns a list
    line_till_5 = 0
    cur_chunk = []
    blank_count = 0
    for line in lines: 
        if line.strip() == "":
            blank_count += 1
        else: 
            blank_count = 0

        if blank_count >= 2:
            if cur_chunk: 
                chunks.append("\n".join(cur_chunk).strip())
                cur_chunk = []
        else: 
            cur_chunk.append(line)
    if cur_chunk:
        chunks.append("\n".join(cur_chunk).strip())

    return chunks
        
    # chunknig it by 500 chars
    def chunk_text(text, chunk_size=200):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])#going by the chunk number in this case 500
                #append the chunks into my chunk list so that each slot has each chunk
        return chunks

    doc = fitz.open(pdf_path)
    for pg_num in range(len(doc)):
        pg = doc.load_page(pg_num)
        full_text += pg.get_text() + "\n"  
    return full_text.strip()

'''