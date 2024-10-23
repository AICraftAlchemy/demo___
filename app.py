import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Set page config
st.set_page_config(
    page_title="LoRA Model Chat",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer once and cache them"""
    max_seq_length = 2048
    dtype = torch.float32  # Explicitly set to float32 for CPU
    load_in_4bit = False  # Disable 4-bit quantization for CPU
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # Replace with your local model path
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="cpu"  # Explicitly set device to CPU
    )
    
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("🤖 LoRA Model Chat Interface")
    
    # Add a sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This is a chat interface for the LoRA-fine-tuned model.
        Enter your message and press Enter or click the Send button to interact.
        Running on CPU only.
        """)
        
        # Add model parameters
        st.markdown("### Model Parameters")
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.5, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=32, max_value=512, value=128, step=32)
        min_p = st.slider("Min P", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Load model and tokenizer
    try:
        model, tokenizer = load_model()
        
        # Create the chat interface
        user_input = st.text_input("Enter your message:", key="user_input")
        
        # Handle user input
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Prepare input for model
            messages = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in st.session_state.chat_history]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cpu")  # Explicitly set to CPU
            
            # Create text streamer
            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            
            # Generate response
            with st.spinner("Generating response..."):
                outputs = model.generate(
                    input_ids=inputs,
                    streamer=text_streamer,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    temperature=temperature,
                    min_p=min_p
                )
                
                # Decode the response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Add model response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        st.markdown("### Chat History")
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
                
        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.markdown("""
        Please make sure:
        1. The model path is correct
        2. All required dependencies are installed
        3. You have sufficient system memory
        """)

if __name__ == "__main__":
    main()
