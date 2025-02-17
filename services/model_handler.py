import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
from agno.tools.pubmed import PubmedTools


# MODEL_PATH = "meta-llama/Llama-3.2-1B"

class ModelHandler:
    def __init__(self):
        """Initialize the model handler"""
        # self.model = None
        # self.tokenizer = None
        self.agent_team = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        # self.model, self.tokenizer = self._load_model()
        researcher = Agent(
            name="Researcher",
            role="You are a research scholar who specializes in autism research.",
            model=Ollama(id="llama3.2:1b"), 
            tools=[ArxivTools(), PubmedTools()],
            instructions=[
                "Be precise and provide just enough information to be useful",
                "You must cite the sources used in your answer.",
            ],
            show_tool_calls=True,
            markdown=True
        )
        summarizer = Agent(
            name="Summarizer",
            role="You are a specialist in summarizing research papers for people without autism knowledge.",
            model=Ollama(id="llama3.2:1b"), 
            instructions=[
                "You must provide just enough information to be useful",
                "You must cite the sources used in your answer.",
                "You must be clear and concise.",
                "You must create an accessible summary."
            ],
            show_tool_calls=True,
            markdown=True
        )
        self.agent_team = Agent(
            name="Agent Team",
            team=[researcher, summarizer],
            model=Ollama(id="llama3.2:1b"),
            show_tool_calls=True,
            markdown=True
        )
        
    
    @staticmethod
    @st.cache_resource
    def _load_model():
        """Load the T5 model and tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None, None
    
    def generate_answer(self, query: str) -> str:
        try:
            response = self.agent_team.run(query)
            return response.content
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer. Please try again or rephrase your question."

    def _get_fallback_response() -> str:
        """Provide a friendly, helpful fallback response"""
        return """I apologize, but I couldn't find enough specific research to properly answer your question. To help you get better information, you could:

            • Ask about specific aspects of autism you're interested in
            • Focus on particular topics like:
            - Early signs and diagnosis
            - Treatment approaches
            - Latest research findings
            - Support strategies

            This will help me provide more detailed, research-backed information that's relevant to your interests."""