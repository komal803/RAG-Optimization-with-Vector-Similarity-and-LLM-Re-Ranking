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
 ```


## 1. Text Cleaning and Chunking  

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
        list1.append(" ".join(list_new[i:i+n]))
    else:
        list1.append(" ".join(list_new[i-10:i+n]))
   ```
