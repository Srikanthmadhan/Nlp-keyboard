import streamlit as st
from collections import defaultdict
import os

# Page configuration
st.set_page_config(
    page_title="Probabilistic Keyboard",
    page_icon="⌨️",
    layout="wide"
)

# Constants
K_SMOOTHING = 1e-6
LAMBDA_TRIGRAM = 0.7
LAMBDA_BIGRAM = 0.2
LAMBDA_UNIGRAM = 0.1
CHARS = 'abcdefghijklmnopqrstuvwxyz '
VOCAB_SIZE = len(CHARS)

# QWERTY keyboard layout
KEYBOARD_LAYOUT = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm']
]

def get_corpus_path():
    """Get corpus path with fallback options"""
    # Try multiple possible locations
    possible_paths = [
        "corpus.txt",
        os.path.join(os.path.dirname(__file__), "corpus.txt"),
        os.path.join(os.getcwd(), "corpus.txt")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def preprocess_corpus(text):
    """Preprocess corpus: lowercase, keep only a-z and space"""
    text = text.lower()
    text = ''.join(c for c in text if c in CHARS)
    text = ' '.join(text.split())  # Normalize spaces
    return text

@st.cache_resource
def build_language_model(corpus_text=None):
    """Build n-gram language model from corpus"""
    
    if corpus_text is None:
        # Try to read from file
        corpus_path = get_corpus_path()
        
        if corpus_path is None:
            st.error("⚠️ Corpus file not found!")
            st.info("Please ensure 'corpus.txt' is in the same directory as this app.")
            
            # Provide option to use sample text
            use_sample = st.checkbox("Use sample corpus for demo?")
            if use_sample:
                corpus = """the quick brown fox jumps over the lazy dog this is a sample text 
                to demonstrate the probabilistic keyboard application when the corpus file is not available 
                you should replace this with your actual training corpus for better predictions"""
            else:
                st.stop()
        else:
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    corpus = f.read()
                st.success(f"✓ Corpus loaded from: {corpus_path}")
            except Exception as e:
                st.error(f"Error reading corpus: {str(e)}")
                st.stop()
    else:
        corpus = corpus_text
    
    # Preprocess
    corpus = preprocess_corpus(corpus)
    
    if len(corpus) == 0:
        st.error("Corpus is empty after preprocessing")
        st.stop()
    
    # Initialize counts
    unigrams = defaultdict(int)
    bigrams = defaultdict(lambda: defaultdict(int))
    trigrams = defaultdict(lambda: defaultdict(int))
    
    # Count unigrams
    for char in corpus:
        unigrams[char] += 1
    
    total_unigrams = sum(unigrams.values())
    
    # Count bigrams
    for i in range(len(corpus) - 1):
        context = corpus[i]
        next_char = corpus[i + 1]
        bigrams[context][next_char] += 1
    
    # Count trigrams
    for i in range(len(corpus) - 2):
        context = corpus[i:i+2]
        next_char = corpus[i + 2]
        trigrams[context][next_char] += 1
    
    return {
        'unigrams': dict(unigrams),
        'bigrams': {k: dict(v) for k, v in bigrams.items()},
        'trigrams': {k: dict(v) for k, v in trigrams.items()},
        'total_unigrams': total_unigrams
    }

def calculate_probabilities(model, context):
    """Calculate probability distribution for next character"""
    
    probs = {}
    
    for char in CHARS:
        # Unigram probability with smoothing
        unigram_count = model['unigrams'].get(char, 0)
        unigram_prob = (unigram_count + K_SMOOTHING) / (model['total_unigrams'] + K_SMOOTHING * VOCAB_SIZE)
        
        if len(context) == 0:
            # No context - use unigram only
            prob = unigram_prob
            
        elif len(context) == 1:
            # Bigram context
            bigram_context = context
            bigram_counts = model['bigrams'].get(bigram_context, {})
            bigram_total = sum(bigram_counts.values())
            bigram_count = bigram_counts.get(char, 0)
            bigram_prob = (bigram_count + K_SMOOTHING) / (bigram_total + K_SMOOTHING * VOCAB_SIZE)
            
            prob = LAMBDA_BIGRAM * bigram_prob + LAMBDA_UNIGRAM * unigram_prob
            
        else:
            # Trigram context (last 2 chars)
            trigram_context = context[-2:]
            trigram_counts = model['trigrams'].get(trigram_context, {})
            trigram_total = sum(trigram_counts.values())
            trigram_count = trigram_counts.get(char, 0)
            trigram_prob = (trigram_count + K_SMOOTHING) / (trigram_total + K_SMOOTHING * VOCAB_SIZE)
            
            # Bigram fallback
            bigram_context = context[-1]
            bigram_counts = model['bigrams'].get(bigram_context, {})
            bigram_total = sum(bigram_counts.values())
            bigram_count = bigram_counts.get(char, 0)
            bigram_prob = (bigram_count + K_SMOOTHING) / (bigram_total + K_SMOOTHING * VOCAB_SIZE)
            
            prob = (LAMBDA_TRIGRAM * trigram_prob + 
                   LAMBDA_BIGRAM * bigram_prob + 
                   LAMBDA_UNIGRAM * unigram_prob)
        
        probs[char] = prob
    
    # Normalize to sum to 100%
    total = sum(probs.values())
    probs = {char: (prob / total) * 100 for char, prob in probs.items()}
    
    return probs

def get_color(prob):
    """Get background color based on probability"""
    intensity = min(prob / 20, 1.0)
    lightness = int(95 - (intensity * 45))
    return f"hsl(200, 70%, {lightness}%)"

def get_text_color(prob):
    """Get text color based on background"""
    intensity = min(prob / 20, 1.0)
    return "white" if intensity > 0.5 else "#1f2937"

# Initialize session state
if 'text' not in st.session_state:
    st.session_state.text = ''
if 'text_input_key' not in st.session_state:
    st.session_state.text_input_key = 0

# Load language model
with st.spinner('Loading language model...'):
    model = build_language_model()

st.success(f"✓ Language model loaded ({len(model['unigrams'])} unique characters, {model['total_unigrams']:,} total chars)")

# Title
st.title("⌨️ Probabilistic Character-Level Keyboard")
st.markdown("*Each key shows the probability (%) of being the next character*")

# Text input area
st.markdown("### Typed Text")

text_input = st.text_area(
    "Type here using your keyboard or click the on-screen keys below:",
    value=st.session_state.text,
    height=120,
    key=f"text_input_{st.session_state.text_input_key}",
    help="You can type directly using your physical keyboard!"
)
if text_input != st.session_state.text:
    st.session_state.text = ''.join(c for c in text_input.lower() if c in CHARS)

# Calculate probabilities
context = st.session_state.text.lower()
probabilities = calculate_probabilities(model, context)

# Display statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Characters Typed", len(st.session_state.text))
with col2:
    if probabilities:
        max_char = max(probabilities, key=probabilities.get)
        display_char = '␣' if max_char == ' ' else max_char
        st.metric("Most Probable", f"'{display_char}' ({probabilities[max_char]:.2f}%)")
with col3:
    context_type = "Unigram" if len(context) == 0 else "Bigram" if len(context) == 1 else "Trigram"
    st.metric("Context Type", context_type)
with col4:
    space_prob = probabilities.get(' ', 0)
    st.metric("Space Probability", f"{space_prob:.2f}%")

st.markdown("---")

# Keyboard display
st.markdown("### On-Screen Keyboard")
st.caption("Click keys below or type directly in the text area above")

# Custom CSS
st.markdown("""
<style>
div[data-testid="stButton"] > button {
    width: 100%;
    height: 70px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    border: 2px solid #cbd5e0;
}
div[data-testid="stButton"] > button:hover {
    border-color: #4299e1;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Create keyboard rows
for row_idx, row in enumerate(KEYBOARD_LAYOUT):
    if row_idx == 0:
        cols = st.columns([0.5] + [1]*10 + [0.5])
        start_col = 1
    elif row_idx == 1:
        cols = st.columns([1] + [1]*9 + [1])
        start_col = 1
    else:
        cols = st.columns([1.5] + [1]*7 + [1.5])
        start_col = 1
    
    for i, char in enumerate(row):
        prob = probabilities.get(char, 0)
        display_char = '␣' if char == ' ' else char.upper()
        
        with cols[start_col + i]:
            button_label = f"{display_char}\n{prob:.2f}%"
            
            if st.button(
                button_label,
                key=f"key_{char}_{row_idx}_{i}",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.text += char
                st.rerun()

# Space bar
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 6, 2])
with col2:
    space_prob = probabilities.get(' ', 0)
    if st.button(
        f"SPACE ({space_prob:.2f}%)", 
        key="space_bar",
        use_container_width=True,
        type="secondary"
    ):
        st.session_state.text += ' '
        st.rerun()

# Control buttons
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 3])

with col2:
    if st.button("⌫ Backspace", use_container_width=True, type="primary"):
        if len(st.session_state.text) > 0:
            st.session_state.text = st.session_state.text[:-1]
            st.rerun()

with col3:
    if st.button("🗑️ Clear All", use_container_width=True, type="primary"):
        st.session_state.text = ''
        st.rerun()

# Footer
st.markdown("---")
with st.expander("ℹ️ About this application"):
    st.markdown("""
    **How it works:**
    - Uses character-level n-gram language model (unigram, bigram, trigram)
    - Applies add-k smoothing (k=1e-6) to handle unseen sequences
    - Interpolates probabilities with weights: λ₃=0.7, λ₂=0.2, λ₁=0.1
    - Displays real-time probability distribution on each key
    
    **Model details:**
    - Corpus: Loaded from `corpus.txt`
    - Vocabulary: a-z and space (27 characters)
    - Context: Uses last 2 characters for trigram model
    - Normalization: All probabilities sum to 100%
    
    **Input methods:**
    - **Physical keyboard**: Type directly in the text area
    - **On-screen keyboard**: Click individual keys
    - Both methods work simultaneously!
    
    **Probability interpretation:**
    - High space probability (~20-30%) reflects natural word boundary frequency in language
    - This is expected behavior, not a bug
    - Darker blue colors indicate higher probability
    
    **Technical notes:**
    - Model is cached for performance (built once at startup)
    - O(27) probability calculation per keystroke
    - Real-time updates with no lag
    """)

# Debug information
if st.checkbox("Show debug information"):
    st.subheader("Debug Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current State:**")
        st.json({
            "Current text": st.session_state.text,
            "Context (last 10)": context[-10:] if len(context) > 10 else context,
            "Context length": len(context),
            "Context type": "Unigram" if len(context) == 0 else "Bigram" if len(context) == 1 else "Trigram"
        })
    
    with col2:
        st.write("**Top 10 Predictions:**")
        top_10 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        for char, prob in top_10:
            display = '␣ (space)' if char == ' ' else f'{char}'
            st.write(f"{display}: {prob:.4f}%")
    
    st.write("**Model Configuration:**")
    st.json({
        "Lambda weights": {
            "Trigram (λ₃)": LAMBDA_TRIGRAM,
            "Bigram (λ₂)": LAMBDA_BIGRAM,
            "Unigram (λ₁)": LAMBDA_UNIGRAM,
            "Sum": LAMBDA_TRIGRAM + LAMBDA_BIGRAM + LAMBDA_UNIGRAM
        },
        "Smoothing (k)": K_SMOOTHING,
        "Vocabulary size": VOCAB_SIZE
    })
