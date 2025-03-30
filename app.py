import streamlit as st
import os
import json
import time
import nltk
import pandas as pd
from docx import Document
from tqdm import tqdm
from hazm import Normalizer, word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Set a persistent directory for NLTK data
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Check and download necessary NLTK components
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)

# Get the API keys from environment variables or use fallback values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "persian-new"  # Updated index name for consistenc

# Fixed chunk parameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 500

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Normalizer
normalizer = Normalizer()

# Helper functions
def load_folder_structure():
    """Load the folder structure from JSON file or create a new one if it doesn't exist"""
    if os.path.exists("folder_structure.json"):
        with open("folder_structure.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"root": []}]  # Default structure if file doesn't exist

def preprocess_text(text):
    """Normalize and tokenize Persian text"""
    text = normalizer.normalize(text)  # Normalize text
    tokens = word_tokenize(text)  # Tokenize words
    return " ".join(tokens)  # Reconstruct the text

def extract_text_from_docx(file):
    """Extract text from a DOCX file"""
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def is_valid_doc(doc):
    """Check if a document is valid"""
    return (
        isinstance(doc["content"], str) and
        len(doc["content"].strip()) > 50
    )

def split_documents(docs):
    """Split documents into chunks using fixed chunk size and overlap"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = []
    count = 1
    for doc in docs:
        for chunk in splitter.split_text(doc["content"]):
            normalized_chunk = preprocess_text(chunk)
            split_docs.append({
                "id": count,
                "content": normalized_chunk, 
                "metadata": doc["metadata"]
            })
            count += 1
    return split_docs

def embed_documents_in_pinecone(docs, index_name):
    """Embed documents into Pinecone"""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    # Load BM25Encoder
    if os.path.exists("full_bm25_values.json"):
        print("Loading BM25Encoder...")
        bm25_encoder = BM25Encoder().load("full_bm25_values.json")
    else:
        # Initialize and save
        all_texts = [doc["content"] for doc in docs]
        bm25_encoder = BM25Encoder().fit(all_texts)
        bm25_encoder.dump("full_bm25_values.json")

    index = pc.Index(index_name)

    all_chunks = [doc["content"] for doc in docs]
    all_metadatas = [doc["metadata"] for doc in docs]

    embeder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # Process documents
    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embeder, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=200
    )

    vectorstore.add_texts(all_chunks, metadatas=all_metadatas)
    return vectorstore

def debug_print_context(inputs):
    """Debug function to print context details."""
    con = inputs.get("context", [])
    context = []
    for doc in con:
        context.append(doc.metadata)
    return inputs

def create_chatbot_retrieval_qa(main_query, additional_note, vs, categories, sub_categories, model_name="gpt-4o"):
    """Modified to handle both main query and additional note with model selection."""
    prompt_template = """
    شما یک دستیار هوشمند و مفید هستید. با استفاده از متن زیر به پرسش مطرح‌شده با دقت، شفافیت، و به صورت کامل پاسخ دهید:
    1. پاسخ را **به زبان فارسی** ارائه دهید.
    2. **جزئیات کامل** را پوشش دهید و اطمینان حاصل کنید که تمام جنبه‌های سؤال به دقت بررسی شده‌اند.
    3. تاریخ‌ها و اطلاعات ارائه‌شده باید **مطابق با متن** باشند. از درج تاریخ‌های نادرست خودداری کنید.
    4. در صورت نیاز به ارجاع به تاریخ، از **نام فایل برای تاریخ دقیق** استفاده کنید.
    5. نام فایل را در مرجع پاسخ بدهید

    **متن:**
    {context}

    **سؤال اصلی:**
    {main_question}

    **یادداشت اضافی:**
    {additional_note}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=OPENAI_API_KEY)

    def filtered_retriever(query):
        filter_dict = {}
        if categories != ['ALL'] and categories != []:
            if categories:
                filter_dict["category"] = {"$in": categories}
        if sub_categories != ['ALL'] and sub_categories != []:
            if sub_categories:
                filter_dict["year"] = {"$in": sub_categories}
        
        return vs.get_relevant_documents(
            query,
            filter=filter_dict
        )

    chain = (
        {
            "context": lambda x: filtered_retriever(x["main_question"]),
            "main_question": lambda x: x["main_question"],
            "additional_note": lambda x: x["additional_note"]
        }
        | RunnablePassthrough(lambda inputs: debug_print_context(inputs))
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain

def initialize_chatbot(alpha=0.3, top_k=60):
    """Initialize the chatbot with Pinecone index and embeddings."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Try to load BM25Encoder, create if doesn't exist
    try:
        bm25_encoder = BM25Encoder().load("full_bm25_values.json")
    except:
        # Create a simple encoder with minimal data if file doesn't exist
        # This is a fallback and should be replaced with proper initialization
        bm25_encoder = BM25Encoder().fit(["placeholder text"])
        bm25_encoder.dump("full_bm25_values.json")

    vectorstore = PineconeHybridSearchRetriever(
        alpha=alpha, 
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=top_k
    )

    return vectorstore

def get_selected_subfolders(selected_folders):
    with open("folder_structure.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    if selected_folders==[]:
        return ['ALL']
    folder_dict = data[0]
    subfolder_list = ['ALL']
    for folder in selected_folders:
        if folder in folder_dict:
            subfolder_list.extend(folder_dict[folder])
    return subfolder_list

# Apply custom CSS for both pages
def apply_custom_css():
    st.markdown("""
        <style>
            body { direction: rtl; text-align: right;}
            h1, h2, h3, h4, h5, h6 { text-align: right; }
            .st-emotion-cache-12fmjuu { display: none;}
            p { font-size:25px !important; }
            .loading-message {
                text-align: center;
                font-size: 20px;
                margin: 20px;
                padding: 20px;
                background-color: #f0f2f6;
                border-radius: 10px;
            }
            .stTextInput input, .stTextArea textarea {
                font-size: 25px !important;
            }
            .st-af {
                font-size: 1.1rem !important;
            }
            .search-params {
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            /* Fix for RTL slider issues */
            .stSlider [data-baseweb="slider"] {
                direction: ltr;
            }
            .stSlider [data-testid="stMarkdownContainer"] {
                text-align: right;
                direction: rtl;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 1px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 60px;
                white-space: pre-wrap;
                font-size: 18px;
                font-weight: 600;
                direction: rtl;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

# Document Upload Page
def document_upload_page():
    st.title("آپلود و پردازش اسناد فارسی")
    st.markdown("""
    این صفحه برای آپلود و پردازش فایل‌های DOCX فارسی و بارگذاری آنها در Pinecone استفاده می‌شود.
    لطفاً ساختار پوشه را انتخاب کرده و اسناد خود را آپلود کنید.
    """)
    
    # Load folder structure
    folder_structure = load_folder_structure()[0]  # Get the first object from the list
    
    # Main layout - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("سازماندهی پوشه اسناد")
        
        # Primary folder selection
        folder_options = list(folder_structure.keys())[1:]  # Skip first item if it's a root
        selected_folder = st.selectbox("انتخاب پوشه اصلی", folder_options)
        
        # New subfolder creation section
        st.subheader("ایجاد زیرپوشه جدید")
        new_subfolder_name = st.text_input("نام زیرپوشه جدید")
        
        if st.button("ایجاد زیرپوشه"):
            if new_subfolder_name and new_subfolder_name.strip():
                # Load the current structure
                with open("folder_structure.json", "r", encoding="utf-8") as f:
                    current_structure = json.load(f)
                
                # Check if the subfolder already exists
                if selected_folder in current_structure[0]:
                    if new_subfolder_name not in current_structure[0][selected_folder]:
                        current_structure[0][selected_folder].append(new_subfolder_name)
                    else:
                        st.warning(f"زیرپوشه {new_subfolder_name} از قبل وجود دارد.")
                else:
                    current_structure[0][selected_folder] = [new_subfolder_name]
                
                # Save the updated structure
                with open("folder_structure.json", "w", encoding="utf-8") as f:
                    json.dump(current_structure, f, ensure_ascii=False, indent=4)
                
                st.success(f"زیرپوشه {new_subfolder_name} با موفقیت اضافه شد.")
                
                # Refresh the page to show the new subfolder
                st.rerun()
            else:
                st.warning("لطفاً نام زیرپوشه را وارد کنید.")
        
        # Subfolder selection (if available)
        subfolder_options = ["None"] + folder_structure[selected_folder]
        selected_subfolder = st.selectbox("انتخاب زیرپوشه", subfolder_options)
        
        if selected_subfolder == "None":
            selected_subfolder = ""
    
    with col2:
        st.subheader("آپلود سند")
        uploaded_file = st.file_uploader("آپلود فایل DOCX فارسی", type=['docx'])
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"فایل: {uploaded_file.name}")
            st.info(f"پوشه انتخاب شده: {selected_folder}/{selected_subfolder}" if selected_subfolder else f"پوشه انتخاب شده: {selected_folder}")
            
            if st.button("پردازش سند"):
                with st.spinner("در حال پردازش سند..."):
                    # Extract text
                    st.info("در حال استخراج متن از سند...")
                    text = extract_text_from_docx(uploaded_file)
                    
                    # Create document with metadata
                    doc = {
                        "content": text,
                        "metadata": {
                            "file_name": uploaded_file.name,
                            "category": selected_folder,
                            "year": selected_subfolder,
                        }
                    }
                    
                    # Chunk documents
                    st.info(f"تقسیم سند به قطعات (اندازه: {CHUNK_SIZE}، همپوشانی: {CHUNK_OVERLAP})...")
                    chunks = split_documents([doc])
                    
                    # Display chunk preview
                    st.success(f"{len(chunks)} قطعه از سند ایجاد شد")
                    
                    if len(chunks) > 0:
                        with st.expander("پیش‌نمایش قطعات"):
                            preview_df = pd.DataFrame({
                                "شناسه": [chunk["id"] for chunk in chunks],
                                "محتوا": [chunk["content"][:200] + "..." for chunk in chunks],
                                "تعداد کاراکترها": [len(chunk["content"]) for chunk in chunks]
                            })
                            st.dataframe(preview_df)
                    
                    # Upload to Pinecone
                    st.info("در حال آپلود قطعات به Pinecone...")
                    try:
                        vectorstore = embed_documents_in_pinecone(chunks, INDEX_NAME)
                        st.success(f"سند با موفقیت پردازش و در شاخص Pinecone آپلود شد: {INDEX_NAME}")
                    except Exception as e:
                        st.error(f"خطا در آپلود به Pinecone: {str(e)}")

# Chatbot Page
def chatbot_page():
    st.markdown("<h1 class='persian-text'>چت‌بات فارسی</h1>", unsafe_allow_html=True)

    # Initialize session state for loading and search parameters
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.3
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 60
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Predefined categories
    with open("folder_structure.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Extract main folder names
    cat = list(data[0].keys()) 

    # Search Parameters Section (collapsible)
    with st.expander("تنظیمات جستجو (پیشرفته)", expanded=False):
        st.markdown("<div class='search-params'>", unsafe_allow_html=True)
        
        # Define callbacks for sliders
        def on_alpha_change():
            st.session_state.vectorstore = None
            
        def on_top_k_change():
            st.session_state.vectorstore = None
        
        # Use two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stSlider">', unsafe_allow_html=True)
            st.session_state.alpha = st.slider(
                "نسبت جستجوی هیبریدی (alpha):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.alpha,
                step=0.1,
                help="مقدار بالاتر به معنای وزن بیشتر برای جستجوی معنایی است. مقدار کمتر وزن بیشتری به جستجوی کلیدواژه می‌دهد.",
                key="alpha_slider",
                on_change=on_alpha_change
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stSlider">', unsafe_allow_html=True)
            st.session_state.top_k = st.slider(
                "تعداد نتایج (top_k):",
                min_value=10,
                max_value=200,
                value=st.session_state.top_k,
                step=10,
                help="تعداد نتایج مرتبطی که از پایگاه داده بازیابی می‌شود.",
                key="top_k_slider",
                on_change=on_top_k_change
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Initialize chatbot if needed
    if st.session_state.vectorstore is None:
        with st.spinner('در حال راه‌اندازی چت‌بات...'):
            try:
                st.session_state.vectorstore = initialize_chatbot(
                    alpha=st.session_state.alpha,
                    top_k=st.session_state.top_k
                )
                st.success(f"پارامترهای جستجو: alpha={st.session_state.alpha}, top_k={st.session_state.top_k}")
            except Exception as e:
                st.error(f"خطا در راه‌اندازی chatbot: {e}")
                return
            
    # Category selections
    categories = st.multiselect(
        "دسته‌بندی را انتخاب کنید:",
        cat,
        default=[]
    )
    
    sub_cat = get_selected_subfolders(categories)

    sub_categories = st.multiselect(
        "زیر دسته‌بندی را انتخاب کنید:",
        sub_cat,
        default=[]
    )
    
    # Model selection - NEW FEATURE
    llm_models = {
        "gpt-4o": "GPT-4o (پیش‌فرض)",
        "gpt-4o-mini": "GPT-4o Mini (سریع‌تر)",
    }
    
    selected_model = st.selectbox(
        "انتخاب مدل زبانی:",
        options=list(llm_models.keys()),
        format_func=lambda x: llm_models[x],
        index=0  # Default to gpt-4o
    )

    # Two separate input fields
    main_query = st.text_area(
        "سؤال اصلی خود را اینجا وارد کنید:",
        height=100
    )

    additional_note = st.text_area(
        "یادداشت اضافی (اختیاری):",
        height=100
    )

    # Submit button
    if st.button("ارسال"):
        response_placeholder = st.empty()

        if not main_query:
            st.warning("لطفاً سؤال اصلی خود را وارد کنید.")
            return

        if not categories and not sub_categories:
            st.warning("لطفاً حداقل یک دسته‌بندی یا زیر دسته‌بندی را انتخاب کنید.")
            return
        
        response_placeholder = st.empty()

        try:
            # Show loading spinner
            with st.spinner('لطفاً صبر کنید...'):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Update progress for vector search
                status_text.text("در حال جستجوی اطلاعات مرتبط...")
                progress_bar.progress(33)
                
                # Create chatbot with the selected model
                chatbot = create_chatbot_retrieval_qa(
                    main_query,
                    additional_note,
                    st.session_state.vectorstore,
                    categories,
                    sub_categories,
                    model_name=selected_model  # Use the selected model
                )
                
                # Update progress for processing
                status_text.text("در حال پردازش اطلاعات...")
                progress_bar.progress(66)
                
                # Get response
                response = chatbot.invoke({
                    "main_question": main_query,
                    "additional_note": additional_note if additional_note else ""
                })
                
                # Update progress for completion
                status_text.text("در حال آماده‌سازی پاسخ...")
                progress_bar.progress(100)
                
                # Clear progress indicators
                time.sleep(0.5)  # Short delay for smooth transition
                progress_bar.empty()
                status_text.empty()

                # Display response
                response_placeholder.markdown("**پاسخ:**")
                response_placeholder.write(response)
                
                # Display model info
                st.info(f"پاسخ با استفاده از مدل {llm_models[selected_model]} تولید شد.")

        except Exception as e:
            st.error(f"خطا در پردازش سوال: {e}")
        finally:
            st.session_state.processing = False

# Main app with tabs
def main():
    st.set_page_config(
        page_title="سامانه پردازش و پرسش و پاسخ فارسی",
        page_icon="🤖",
        layout="wide",
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Create tabs
    tab1, tab2 = st.tabs(["چت‌بات فارسی  "," "+ "  آپلود و پردازش اسناد"])
    
    with tab1:
        chatbot_page()
        
    with tab2:
        document_upload_page()

if __name__ == "__main__":
    main()