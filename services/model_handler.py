import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
from agno.tools.pubmed import PubmedTools

MODEL_PATH = "meta-llama/Llama-3.2-1B"

class ModelHandler:
    def __init__(self):
        """Initialize the model handler"""
        self.model = None
        self.tokenizer = None
        self.translator = None
        self.researcher = None
        self.summarizer = None
        self.presenter = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        self.model, self.tokenizer = self._load_model()
        self.translator = Agent(
            name="Translator",
            role="You will translate the query to English",
            model=Ollama(id="llama3.2:1b"),
            goal="Translate to English",
            instructions=[
                "Translate the query to English"
            ]
        ) 
        
        self.researcher = Agent(
            name="Researcher",
            role="You are a research scholar who specializes in autism research.",
            model=Ollama(id="llama3.2:1b"), 
            tools=[ArxivTools(), PubmedTools()],
            instructions=[
                "You need to understand the context of the question to provide the best answer based on your tools."
                "Be precise and provide just enough information to be useful",
                "You must cite the sources used in your answer."
                "You must create an accessible summary.",
                "The content must be for people without autism knowledge.",
                "Focus in the main findings of the paper taking in consideration the question.",
                "The answer must be brief."
            ],
            show_tool_calls=True,
        )
        self.summarizer = Agent(
            name="Summarizer",
            role="You are a specialist in summarizing research papers for people without autism knowledge.",
            model=Ollama(id="llama3.2:1b"), 
            instructions=[
                "You must provide just enough information to be useful",
                "You must cite the sources used in your answer.",
                "You must be clear and concise.",
                "You must create an accessible summary.",
                "The content must be for people without autism knowledge.",
                "Focus in the main findings of the paper taking in consideration the question.",
                "The answer must be brief."
                "Remove everything related to the run itself like: 'Running: transfer_', just use plain text",
                "You must use the language provided by the user to present the results.",
                "Add references to the sources used in the answer.",
                "Add emojis to make the presentation more interactive."
                "Translaste the answer to Portuguese."
            ],
            show_tool_calls=True,
            markdown=True,
            add_references=True,
        )
        
        self.presenter = Agent(
            name="Presenter",
            role="You are a professional researcher who presents the results of the research.",
            model=Ollama(id="llama3.2:1b"),
            instructions=[
                "You are multilingual",
                "You must present the results in a clear and concise manner.",
                "Clenaup the presentation to make it more readable.",
                "Remove unnecessary information.",
                "Remove everything related to the run itself like: 'Running: transfer_', just use plain text",
                "You must use the language provided by the user to present the results.",
                "Add references to the sources used in the answer.",
                "Add emojis to make the presentation more interactive."
                "Translaste the answer to Portuguese."
            ],
            add_references=True,
        )
        
    
    @staticmethod
    @st.cache_resource
    @st.cache_data
    def _load_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None, None
    
    def generate_answer(self, query: str) -> str:
        try:
            translator = self.translator.run(query, stream=False)
            logging.info(f"Translated query")
            research = self.researcher.run(translator.content, stream=False)
            logging.info(f"Generated research")
            summary = self.summarizer.run(research.content, stream=False)
            logging.info(f"Generated summary")
            presentation = self.presenter.run(summary.content, stream=False)
            logging.info(f"Generated presentation")
            
            if not presentation.content:
                return self._get_fallback_response()
            return presentation.content
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return self._get_fallback_response()
    
    @staticmethod
    def _get_fallback_response() -> str:
        """Provide a friendly, helpful fallback response"""
        return """
            Peço descula, mas encontrei um erro ao gerar a resposta. Tente novamente ou refaça a sua pergunta.
        """