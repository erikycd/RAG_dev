import os
import sys
from . import keys
os.environ["OPENAI_API_KEY"] = keys.OPENAI_API_KEY
from src.config import RAGConfig
from src.indexing.document_processor import DocumentProcessor
from src.generation.gpt_rag import GPTRAG
from src.generation.local_rag import LocalRAG

def main(model):
    """
    Main function to run the RAG system.
    Args:
        model (str): Model to use ('GPT' or 'Local').
    """
    try:
        # Initialize configuration
        config = RAGConfig()
        # Load and process documents
        print("Loading documents...")
        doc_processor = DocumentProcessor(config)
        documents = doc_processor.load_documents(
            './data/raw/Article_1.pdf'
        )
        # Initialize selected model
        if model.upper() == 'GPT':
            print("Initializing GPT model...")
            rag = GPTRAG(config, documents)
            print("GPT model ready.")
        elif model.upper() == 'LOCAL':
            print("Initializing local model...")
            rag = LocalRAG(config, documents)
            print("Local model ready.")
        else:
            print(f"Model '{model}' not recognized. Using GPT by default.")
            rag = GPTRAG(config, documents)
        print("\n" + "="*50)
        print(f"RAG system initialized in {model.upper()} mode.")
        print("Write your query or 'quit' to exit.")
        print("Write 'help' to see more commands.")
        print("="*50 + "\n")
        # Main interaction loop
        while True:
            try:
                # Get user input
                query = input("\nUser: ").strip()
                # Handle special commands
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'salir']:
                    print("\nAsistant: Bye!")
                    break
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- help: Show this help")
                    print("- quit/exit/salir: Terminate the session")
                    print("- new: Restart the conversation")
                    continue
                if query.lower() == 'new':
                    print("\nNew conversation started")
                    continue
                # Generate and show response
                print("\nThinking...")
                response = rag.generate_response(query)
                print(f"\nAssistant: {response}")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                continue
    except Exception as e:
        print(f"\nCritical error initializing system: {str(e)}")
        return

if __name__ == "__main__":
    """
    Main
    Args:
        python main.py GPT
        python -m src.main GPT

        python main.py LOCAL
        python -m src.main LOCAL
    """
    model = str(sys.argv[1])
    main(model)
