# RAG Optimization with Vector Similarity and LLM-Re-Ranking


## Abstract  
This project introduces a novel hybrid approach to enhance Retrieval-Augmented Generation (RAG) systems by integrating vector similarity and question-driven re-ranking using Large Language Models (LLMs). Utilizing SBERT for initial similarity calculations and a fine-tuned BART model for question-based re-ranking, this methodology significantly improves the accuracy and contextual relevance of retrieved chunks, offering a scalable solution for intelligent information retrieval systems.

## Introduction  
Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval by combining retrieval with generative capabilities. However, traditional methods relying on Manhattan, Euclidean, or cosine similarity often fail to retrieve contextually relevant data, especially for complex queries.  

This project tackles this limitation by adopting a hybrid approach:
- **Vector Similarity**: Used for initial chunk ranking.  
- **LLM-Based Question Generation**: Enhances contextual understanding through question-driven re-ranking.  

This method bridges the gap in traditional RAG pipelines, yielding more precise and relevant outputs.

## Methodology  
The hybrid methodology comprises the following steps:

1. **Chunking Data**:  
   - Text data is divided into chunks of 50–100 tokens.  
   - Chunks are stored as a DataFrame for efficient processing.
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


2. **Encoding and Similarity Calculation**:  
   - User queries and text chunks are encoded using **SBERT**.  
   - **Cosine similarity** is computed to rank chunks.  
   - Top 25 chunks are selected for further processing.

3. **Question Generation**:  
   - Fine-tuned **BART** model, trained on the `anon-betterbench/betterbench-b1-all-questions` dataset, generates three questions per chunk.  
   - Chunks are divided into structured groups (10+10+5) for question generation.

4. **Re-Ranking**:  
   - Chunks are re-ranked based on question similarity scores.  
   - Results are sorted in descending order of relevance.

5. **Final Selection**:  
   - Top 20 rows from the re-ranked DataFrame are analyzed.  
   - Value counts are used to identify the chunk most relevant to the query.

## Results  
The hybrid approach achieves:  
- **Improved Retrieval Accuracy**: Enhanced precision in chunk selection using SBERT.  
- **Contextual Understanding**: Fine-tuned BART model improves chunk ranking through question-driven re-ranking.  
- **Dynamic Evaluation**: A combination of vector similarity and question re-ranking refines the RAG pipeline.

## Challenges  
- Computational cost of fine-tuning transformer-based models.  
- Complexity of handling and re-ranking large datasets.

## Technologies Used  
- **Language Models**: SBERT, Fine-tuned BART  
- **Similarity Metrics**: Cosine Similarity  
- **Data Processing**: Python, Pandas  
- **Frameworks**: Hugging Face Transformers  

## Future Work  
- Explore optimization techniques to reduce computational cost.  
- Apply the hybrid methodology to various domains such as legal, healthcare, and finance.  
- Develop strategies for handling larger datasets more efficiently.  

## Acknowledgements  
This project is inspired by the advancements in transformer-based models and retrieval systems. The contributions of **Hugging Face** for pre-trained models and datasets like `anon-betterbench` are greatly appreciated.
