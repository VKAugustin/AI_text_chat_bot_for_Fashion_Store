from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.palm import PaLM
from llama_index import ServiceContext
from llama_index import StorageContext
import os


class RAGPaLMQuery:
    def __init__(self):
       
        if not os.path.exists("data"):
            os.makedirs("data")

       
        self.documents = SimpleDirectoryReader("./data").load_data()

      
        os.environ['GOOGLE_API_KEY'] = 'AIzaSyA88VrhnSqYY0gWO-SewogFg9WUvSm5SBw'

       
        self.llm = PaLM()
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

       
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, chunk_size=800, chunk_overlap=20)

      
        self.index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context)

        self.index.storage_context.persist()
        

       
        self.query_engine = self.index.as_chat_engine()

    def query_response(self, query):
       
        response = self.query_engine.chat(query)
        return response
        
        