# RAG Optimization with Vector Similarity and LLM Re-Ranking

## Abstract  
This project introduces a hybrid approach to enhance Retrieval-Augmented Generation (RAG) systems by integrating vector similarity and question-driven re-ranking using Large Language Models (LLMs). By utilizing SBERT for initial similarity calculations and a fine-tuned BART model for question-based re-ranking, the project achieves improved accuracy and contextual relevance in retrieved chunks.

---

## Features  
- **Text Cleaning and Chunking**: Prepares raw text for processing by dividing it into overlapping chunks.  
- **Vector Similarity**: Calculates similarity between user queries and text chunks using SBERT.  
- **Question Generation**: Leverages a fine-tuned BART model to generate questions for re-ranking retrieved chunks.  
- **Dynamic Re-Ranking**: Refines ranking based on similarity between generated questions and the query.  
- **Persistent Storage**: Integrates ChromaDB for storing and querying chunks efficiently.

---

## Installation  
1. Clone the repository:  
   ```python
   git clone https://github.com/yourusername/rag-optimization.git
   cd rag-optimization
 ``` ```  
## Text Cleaning and Chunking

The raw text is cleaned to remove unwanted characters, converted to lowercase, and split into overlapping chunks for further processing.

```python
import re

def clean_text(text):
    """Cleans text data by removing punctuation, numbers, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', "", text)  # Remove punctuation
    text = re.sub(r'\d+', "", text)      # Remove numbers
    return text

# Splitting the text into overlapping chunks
text = "Your text data here"
list_new = text.split(" ")
list1 = []
n = 100  # Chunk size
for i in range(0, len(list_new) - n, n):
    if i == 0:
        list1.append(" ".join(list_new[i:i + n]))
    else:
        list1.append(" ".join(list_new[i - 10:i + n]))
 ``` 
## 2. Vector Similarity Using SBERT
SBERT is used to calculate the similarity between the query and each chunk of text, ranking the chunks by their relevance.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
query = "Your query here"
query_embedding = model.encode(query)

# Calculate similarity for each chunk
sim_scores = []
for chunk in list1:
    chunk_embedding = model.encode(chunk)
    similarity = np.dot(query_embedding, chunk_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
    )
    sim_scores.append(similarity)

# Create a DataFrame with chunks and similarity scores
df = pd.DataFrame({'Chunks': list1, 'Similarity': sim_scores})
df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
```

## 3. Question Generation Using BART
The fine-tuned BART model generates questions for the top chunks to improve contextual understanding and relevance.

from transformers import BartForConditionalGeneration, BartTokenizer

```python
# Load the fine-tuned BART model and tokenizer
model_name = './fine-tuned-bart-model-path'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def process_in_batches(texts, batch_size):
    questions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=2, early_stopping=True)
        questions.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    return questions

top_chunks = df.head(25)['Chunks'].tolist()
questions = process_in_batches(top_chunks, batch_size=10)
df['Questions'] = questions

```
## 4. Storage with ChromaDB
ChromaDB is used to store text chunks and their embeddings persistently for querying.

```python
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma-db-path")
collection = chroma_client.get_or_create_collection(name="text_chunks")

# Add chunks to the collection
ids = [str(i) for i in range(len(list1))]
collection.add(documents=list1, ids=ids)

# Query the collection
results = collection.query(query_texts=[query], n_results=5)
print(results)
```


## Results
Improved Accuracy: Enhanced precision in retrieving relevant chunks.
Contextual Understanding: Question-driven re-ranking ensures relevance.
Efficient Storage: ChromaDB allows for persistent and scalable data handling.
## Future Work
Optimize computational costs for model fine-tuning and re-ranking.
Extend the system to handle larger datasets and domains like healthcare and finance.
Explore integration with advanced transformer models.
