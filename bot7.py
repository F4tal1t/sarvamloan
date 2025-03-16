import os
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIGURATION =====
token = os.environ["OPENAI_API_KEY"]  # Set in your environment
endpoint = "https://models.inference.ai.azure.com"
embedding_model = "text-embedding-3-small"
chat_model = "gpt-4o"

# Sarvam AI Translate API configuration
sarvam_url = "https://api.sarvam.ai/translate"
sarvam_api_key = "<your-sarvam-api-key>"  # Replace with your Sarvam API key

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# ===== TRANSLATION FUNCTIONS =====
def translate_text(text, source_lang="auto", target_lang="en-IN"):
    """
    Translate text using Sarvam AI Translate API.
    """
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Female",  # Optional
        "mode": "formal",  # Optional
        "enable_preprocessing": False,  # Optional
        "output_script": "roman",  # Optional
        "numerals_format": "international"  # Optional
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": sarvam_api_key
    }

    response = requests.post(sarvam_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("translated_text")
    else:
        print("Translation error:", response.status_code, response.json())
        return None

def detect_language(text):
    """
    Detect the language of the input text using Sarvam AI Translate API.
    """
    payload = {
        "input": text,
        "source_language_code": "auto",
        "target_language_code": "en-IN"  # Target language doesn't matter for detection
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": sarvam_api_key
    }

    response = requests.post(sarvam_url, json=payload, headers=headers)
    if response.status_code == 200:
        # Sarvam AI returns the detected language in the response
        return response.json().get("source_language_code")
    else:
        print("Language detection error:", response.status_code, response.json())
        return None

# ===== LOAD THE DATASETS =====
# Loan Dataset
loan_df = pd.read_csv(r"C:\Users\anike\OneDrive\Desktop\GitHub\Python\practice\bank_loan_requirements_detailed.csv")
# Financial Literacy Dataset
finlit_df = pd.read_csv(r"C:\Users\anike\OneDrive\Desktop\GitHub\Python\practice\financial_literacy_tips.csv")

# ===== CREATE DOCUMENTS =====
def create_loan_documents(df):
    documents = []
    for idx, row in df.iterrows():
        doc = (
            f"Bank Name: {row['Bank Name']}\n"
            f"Loan Type: {row['Loan Type']}\n"
            f"Interest Rate (%): {row['Interest Rate (%)']}%\n"
            f"Loan Tenure: {row['Loan Tenure (Min years)']} to {row['Loan Tenure (Max years)']} years\n"
            f"Minimum Loan Amount: ₹{row['Minimum Loan Amount']}\n"
            f"Maximum Loan Amount: ₹{row['Maximum Loan Amount']}\n"
            f"Eligibility Criteria: {row['Eligibility Criteria']}\n"
            f"Minimum Credit Score: {row['Minimum Credit Score']}\n"
            f"Required Documents: {row['Required Documents']}\n"
            f"Processing Fees: {row['Processing Fees']}\n"
            f"Prepayment & Late Payment Charges: {row['Prepayment & Late Payment Charges']}"
        )
        documents.append(doc)
    return documents

def create_finlit_documents(df):
    documents = []
    for idx, row in df.iterrows():
        doc = f"Category: {row['Category']}. Tip: {row['Tip']}."
        documents.append(doc)
    return documents

loan_documents = create_loan_documents(loan_df)
finlit_documents = create_finlit_documents(finlit_df)

# ===== EMBEDDING FUNCTIONS =====
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=embedding_model
    )
    return response.data[0].embedding

def compute_and_save_embeddings(documents, filename):
    embeddings = []
    for i, doc in enumerate(documents):
        print(f"Embedding document {i+1}/{len(documents)}")
        embedding = get_embedding(doc)
        embeddings.append(embedding)

    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {filename}")
    return embeddings

# ===== LOAD OR COMPUTE EMBEDDINGS =====
loan_embeddings_file = r"C:\Users\anike\OneDrive\Desktop\GitHub\Python\practice\loan_dataset_embeddings_openai.pkl"
finlit_embeddings_file = r"C:\Users\anike\OneDrive\Desktop\GitHub\Python\practice\financial_literacy_embeddings_openai.pkl"

try:
    with open(loan_embeddings_file, "rb") as f:
        loan_embeddings = pickle.load(f)
    print("Loaded loan embeddings.")
except FileNotFoundError:
    print("Loan embeddings not found. Generating...")
    loan_embeddings = compute_and_save_embeddings(loan_documents, loan_embeddings_file)

try:
    with open(finlit_embeddings_file, "rb") as f:
        finlit_embeddings = pickle.load(f)
    print("Loaded financial literacy embeddings.")
except FileNotFoundError:
    print("Financial literacy embeddings not found. Generating...")
    finlit_embeddings = compute_and_save_embeddings(finlit_documents, finlit_embeddings_file)

# ===== EMBEDDING FUNCTION FOR QUERY =====
def get_query_embedding(query):
    response = client.embeddings.create(
        input=[query],
        model=embedding_model
    )
    return response.data[0].embedding

# ===== SIMILARITY SEARCH ACROSS BOTH DATASETS =====
def find_most_relevant_documents(query, top_k=3):
    query_embedding = get_query_embedding(query)
    
    # Loan similarity
    loan_similarities = cosine_similarity([query_embedding], loan_embeddings)[0]
    loan_top_indices = np.argsort(loan_similarities)[::-1][:top_k]
    loan_top_docs = [(loan_similarities[i], loan_documents[i]) for i in loan_top_indices]
    
    # Financial literacy similarity
    finlit_similarities = cosine_similarity([query_embedding], finlit_embeddings)[0]
    finlit_top_indices = np.argsort(finlit_similarities)[::-1][:top_k]
    finlit_top_docs = [(finlit_similarities[i], finlit_documents[i]) for i in finlit_top_indices]
    
    # Combine & sort by similarity score
    combined_docs = loan_top_docs + finlit_top_docs
    combined_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top_k most relevant documents
    top_results = [doc for score, doc in combined_docs[:top_k]]
    return top_results

# ===== USER INFO DICTIONARY =====
user_info = {}

# ===== FUNCTION TO UPDATE USER INFO =====
def update_user_info():
    print("\nLet's update your personal info! Leave blank to skip a field.\n")
    
    name = input("Enter your name: ").strip()
    if name: user_info['Name'] = name

    age = input("Enter your age: ").strip()
    if age: user_info['Age'] = age

    occupation = input("Enter your occupation: ").strip()
    if occupation: user_info['Occupation'] = occupation

    annual_income = input("Enter your annual income (₹): ").strip()
    if annual_income: user_info['Annual Income'] = annual_income

    credit_score = input("Enter your credit score: ").strip()
    if credit_score: user_info['Credit Score'] = credit_score

    loan_purpose = input("What is the purpose of the loan you're interested in?: ").strip()
    if loan_purpose: user_info['Loan Purpose'] = loan_purpose

    print("\nYour information has been updated!\n")
    print("Current User Info:")
    for key, value in user_info.items():
        print(f"{key}: {value}")
    print()

# ===== GPT-4o ANSWER GENERATION FUNCTION =====
def generate_answer(context, query):
    system_prompt = "You are a helpful assistant providing information about bank loans and financial literacy tips from embedded data."

    # Combine user info into a string
    user_info_context = ""
    if user_info:
        user_info_context = "User Information:\n" + "\n".join([f"{k}: {v}" for k, v in user_info.items()]) + "\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_info_context}Use the following information to answer the user's query:\n\n{context}\n\nUser query: {query}\n\nAnswer:"}
    ]
    
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()

# ===== MAIN CHAT FUNCTION =====
def chat():
    print("\nWelcome to the Smart Finance Chatbot!")
    print("Ask your questions about loans or financial literacy.")
    print("Type 'update info' to enter or update your personal info.")
    print("Type 'exit' to quit.\n")

    # Ask user if they want to enter info at the start
    first_time = input("Would you like to enter your information now? (yes/no): ").strip().lower()
    if first_time == 'yes':
        update_user_info()
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        elif query.lower() == 'update info':
            update_user_info()
            continue
        
        # Detect the user's input language
        source_lang = detect_language(query)
        if not source_lang:
            print("Chatbot: Language detection failed. Please try again.")
            continue

        # Translate user input to English
        translated_query = translate_text(query, source_lang=source_lang, target_lang="en-IN")
        if not translated_query:
            print("Chatbot: Translation failed. Please try again.")
            continue
        
        # Get top relevant documents from both datasets
        top_docs = find_most_relevant_documents(translated_query)
        
        # Combine into context
        context = "\n\n".join(top_docs)
        
        # Generate an answer from GPT-4o
        answer = generate_answer(context, translated_query)
        
        # Translate the answer back to the user's language
        translated_answer = translate_text(answer, source_lang="en-IN", target_lang=source_lang)
        if not translated_answer:
            translated_answer = answer  # Fallback to English if translation fails
        
        print(f"\nChatbot: {translated_answer}\n")

# ===== RUN THE CHATBOT =====
if __name__ == "__main__":
    chat()