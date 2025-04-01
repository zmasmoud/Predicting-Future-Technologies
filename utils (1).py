import faiss
import json
import pypdf
import os 
from tqdm import tqdm

class ChunkLoader:
    @staticmethod
    def load_chunks(file_path):
        document_chunks = []
        with open(file_path, 'r') as f:
            for line in f:
                chunk = json.loads(line)["chunk"]
                document_chunks.append(chunk)
        return document_chunks

class RAGEncoder:
    def __init__(self, rag_encoder_model, rag_encoder_tokenizer):
        self.rag_encoder_model = rag_encoder_model
        self.rag_encoder_tokenizer = rag_encoder_tokenizer

    def encode_text(self, text):
        # inputs = self.rag_encoder_tokenizer(question, return_tensors='pt')
        # with torch.no_grad():
        #     text_embedding = self.rag_encoder_model(**inputs).pooler_output.numpy()
        text_embedding = self.rag_encoder_model.encode(text)
        return text_embedding

def retrieve_relevant_docs(question, index, chunks, question_encoder, topk=3, normalize=True):
    question_embedding = question_encoder.encode_text(question)
    print(type(question_embedding))
    question_embedding = question_embedding.reshape(1, -1) if question_embedding.ndim == 1 else question_embedding
    #normalize the question embedding so that inner product between question and chunk = cosine similarity
    if normalize: faiss.normalize_L2(question_embedding)
    distances, indices = index.search(question_embedding, topk)
    return [(chunks[idx], distances[0][j]) for j, idx in enumerate(indices[0])]

def read_pdf(file_path):
  reader = pypdf.PdfReader(file_path)
  text = ""
  for page in reader.pages:
    text += page.extract_text()
  return text

def get_pdf_chunks(text, tokenizer, max_chunk_size, overlap):
  chunks = []
  tokens = tokenizer.encode(text, add_special_tokens=False)
  for i in range(0, len(tokens), max_chunk_size - overlap):
    chunk_tokens = tokens[i:i+max_chunk_size]
    chunk_text = tokenizer.decode(chunk_tokens)
    chunks.append(chunk_text)
  return chunks

def get_all_chunks(folder_path, tokenizer, max_chunk_size, overlap):
  all_chunks = []
  for file_name in tqdm(os.listdir(folder_path), desc="reading folder..."):
    text = read_pdf(folder_path + "/" + file_name)
    chunks = get_pdf_chunks(text, tokenizer, max_chunk_size, overlap)
    all_chunks += chunks
  return all_chunks