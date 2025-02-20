import streamlit as st
import logging
from services.model_handler import ModelHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AutismResearchApp:
    def __init__(self):
        """Initialize the application components"""
        self.model_handler = ModelHandler()
    
    def _setup_streamlit(self):
        """Setup Streamlit UI components"""
        st.image("./cover.jpg")
        st.title("🧩 Além do Espectro 🧠✨")
        st.subheader("Tudo o que você precisa saber além dos rotulos e explorando a riqueza das neurodivergências")
        st.markdown("""
            Pergunte o que quiser e eu vou analisar os últimos artigos científicos e fornecer uma resposta baseada em evidências.
        """)
    
    def run(self):
        """Run the main application loop"""
        self._setup_streamlit()
        
        # Initialize session state for papers
        if 'papers' not in st.session_state:
            st.session_state.papers = []
        
        # Get user query
        col1, col2 = st.columns(2, vertical_alignment="bottom", gap="small")

        query = col1.text_input("O que você precisa saber?")
        if col2.button("Enviar"):
            # Show status while processing
            with st.status("Processando sua Pergunta...") as status:
                status.write("🔍 Buscando papers de pesquisa relevantes...")
                status.write("📚 Analisando papers de pesquisa...")
                status.write("✍️ Gerando resposta...")
                answer = self.model_handler.generate_answer(query)
                
                status.write("✨ Resposta gerada! Exibindo resultados...")

            st.success("✅ Resposta gerada com base nos artigos de pesquisa encontrados.")

            
            st.markdown("### Resposta")
            st.markdown(answer)

def main():
    app = AutismResearchApp()
    app.run()

if __name__ == "__main__":
    main()