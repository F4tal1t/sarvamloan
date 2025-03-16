import os
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from appwrite.client import Client
from appwrite.services.databases import Databases
import random
import string
from fastapi import FastAPI

app = FastAPI()

# ===== CONFIGURATION =====
token = os.environ["OPENAI_API_KEY"]  # Set in your environment
endpoint = "https://models.inference.ai.azure.com"
embedding_model = "text-embedding-3-small"
chat_model = "gpt-4o"

# Sarvam AI Translate API configuration
sarvam_url = "https://api.sarvam.ai/translate"
sarvam_api_key = "396f1306-e883-417c-aeb3-44f0d375f0f0"  # Replace with your Sarvam API key

# Appwrite configuration
appwrite_endpoint = 'https://cloud.appwrite.io/v1'  # Your Appwrite endpoint
appwrite_project_id = '67d57daf002484b47c54'  # Your Appwrite project ID
appwrite_api_key = 'standard_2d92766ae05f91ad5f80f7c6bda5c390cefc89f0e35faa665147cc76d1dafc8003436e19a17892c93aa4164a2b2ff600ff3745b843891dac00574cc58a37a4e09fa730aac9c682cbad5fa6d8499612fbe4a62a86cad91a2d63ac885c0baed27d839e45289e95971ce659fffd49559fac678dafa10bb0bd901d98cc800c5d10a2'  # Your Appwrite API key
appwrite_database_id = '67d5ccfc0007c41248bc'  # Your Appwrite database ID
appwrite_chatbot_logs_collection_id = '67d5dcee0034df0653e5'  # Your Appwrite collection ID for chat logs

# Initialize OpenAI client
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Initialize Appwrite client
appwrite_client = Client()
appwrite_client.set_endpoint(appwrite_endpoint)
appwrite_client.set_project(appwrite_project_id)
appwrite_client.set_key(appwrite_api_key)

# Initialize Appwrite Databases service
databases = Databases(appwrite_client)

# ===== TRANSLATION FUNCTIONS =====
def translate_text(text, source_lang="auto", target_lang="en-IN", output_script="fully-native"):
    """
    Translate text using Sarvam AI Translate API.
    """
    # Validate input length (max 1000 characters)
    if len(text) > 1000:
        print("Input text exceeds maximum length (1000 characters). Truncating...")
        text = text[:1000]  # Truncate to 1000 characters

    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Female",  # Optional
        "mode": "formal",  # Optional
        "enable_preprocessing": False,  # Optional
        "output_script": output_script,  # Use native script
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

# ===== POSITIVE AND NEGATIVE WORD LISTS =====
positive_words = ["yes", "ok", "sure", "yup", "yeah", "yep", "of course", "absolutely"]
negative_words = ["no", "nah", "nope", "cancel", "not", "never"]

def is_positive_response(text):
    """
    Check if the translated text contains a positive response.
    """
    return any(word in text.lower() for word in positive_words)

def is_negative_response(text):
    """
    Check if the translated text contains a negative response.
    """
    return any(word in text.lower() for word in negative_words)

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
def update_user_info(preferred_language):
    """
    Update user information in the user's preferred language and native script.
    """
    # Translate prompts into the user's preferred language and native script
    prompts = {
        "name": "Enter your name: ",
        "age": "Enter your age: ",
        "occupation": "Enter your occupation: ",
        "annual_income": "Enter your annual income (₹): ",
        "credit_score": "Enter your credit score: ",
        "loan_purpose": "What is the purpose of the loan you're interested in?: "
    }

    translated_prompts = {}
    for key, prompt in prompts.items():
        translated_prompt = translate_text(prompt, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native")
        if not translated_prompt:
            translated_prompt = prompt  # Fallback to English if translation fails
        translated_prompts[key] = translated_prompt

    print("\nLet's update your personal info! Leave blank to skip a field.\n")
    
    name = input(translated_prompts["name"]).strip()
    if name: user_info['Name'] = name

    age = input(translated_prompts["age"]).strip()
    if age: user_info['Age'] = age

    occupation = input(translated_prompts["occupation"]).strip()
    if occupation: user_info['Occupation'] = occupation

    annual_income = input(translated_prompts["annual_income"]).strip()
    if annual_income: user_info['Annual Income'] = annual_income

    credit_score = input(translated_prompts["credit_score"]).strip()
    if credit_score: user_info['Credit Score'] = credit_score

    loan_purpose = input(translated_prompts["loan_purpose"]).strip()
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

# ===== CHAT LOGGING FUNCTION =====
def generate_short_id(length=20):
    """
    Generate a random string of fixed length.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def log_chat_interaction(user_id, query, response, language):
    """
    Log the chat interaction in the Appwrite chatbot_logs collection.
    """
    try:
        # Generate a short unique chat_id
        chat_id = generate_short_id()

        # Truncate the response to 255 characters if it exceeds the limit
        if len(response) > 255:
            response = response[:255]  # Truncate to 255 characters

        # Create a document in the chatbot_logs collection
        databases.create_document(
            database_id=appwrite_database_id,
            collection_id=appwrite_chatbot_logs_collection_id,
            document_id='unique()',  # Auto-generate document ID
            data={
                'chat_id': chat_id,  # Add chat_id
                'user_id': user_id,
                'query': query,
                'response': response,  # Truncated response
                'language': language,
                'timestamp': datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
            }
        )
        print("Chat interaction logged successfully.")
    except Exception as e:
        print(f"Failed to log chat interaction: {e}")

# ===== MAIN CHAT FUNCTION =====
def chat():
    # Welcome message in English (will be translated later)
    welcome_message = "Welcome to the Smart Finance Chatbot!"
    instructions = "Ask your questions about loans or financial literacy."
    update_info_prompt = "Type 'update info' to enter or update your personal info."
    exit_prompt = "Type 'exit' to quit."

    # Language selection prompt
    language_prompt = "Please choose your preferred language:"
    language_options = [
        "1. English (en-IN)",
        "2. हिंदी (hi-IN)",
        "3. বাংলা (bn-IN)",
        "4. ગુજરાતી (gu-IN)",
        "5. ಕನ್ನಡ (kn-IN)",
        "6. മലയാളം (ml-IN)",
        "7. मराठी (mr-IN)",
        "8. ଓଡିଆ (od-IN)",
        "9. ਪੰਜਾਬੀ (pa-IN)",
        "10. தமிழ் (ta-IN)",
        "11. తెలుగు (te-IN)"
    ]

    # Map choice to language code
    language_map = {
        "1": "en-IN",
        "2": "hi-IN",
        "3": "bn-IN",
        "4": "gu-IN",
        "5": "kn-IN",
        "6": "ml-IN",
        "7": "mr-IN",
        "8": "od-IN",
        "9": "pa-IN",
        "10": "ta-IN",
        "11": "te-IN"
    }

    # Prompt user for preferred language
    print(language_prompt)
    for option in language_options:
        print(option)
    language_choice = input("Enter the number corresponding to your preferred language: ").strip()

    # Set preferred language
    preferred_language = language_map.get(language_choice, "en-IN")  # Default to English if invalid choice

    # Translate all English strings into the preferred language
    welcome_message_translated = translate_text(welcome_message, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or welcome_message
    instructions_translated = translate_text(instructions, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or instructions
    update_info_prompt_translated = translate_text(update_info_prompt, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or update_info_prompt
    exit_prompt_translated = translate_text(exit_prompt, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or exit_prompt

    # Display translated welcome message and instructions
    print(f"\n{welcome_message_translated}")
    print(f"{instructions_translated}")
    print(f"{update_info_prompt_translated}")
    print(f"{exit_prompt_translated}\n")

    # Ask user if they want to enter info at the start
    first_time_prompt = translate_text("Would you like to enter your information now? (yes/no): ", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native")
    if not first_time_prompt:
        first_time_prompt = "Would you like to enter your information now? (yes/no): "  # Fallback to English

    user_response = input(first_time_prompt).strip()
    translated_response = translate_text(user_response, source_lang=preferred_language, target_lang="en-IN")

    if translated_response and is_positive_response(translated_response):
        update_user_info(preferred_language)
    else:
        skip_message = translate_text("Skipping user info update.", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or "Skipping user info update."
        print(skip_message)
    
    while True:
        query_prompt = translate_text("You: ", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or "You: "
        query = input(query_prompt).strip()
        
        if query.lower() in ['exit', 'quit']:
            goodbye_message = translate_text("Chatbot: Goodbye!", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or "Chatbot: Goodbye!"
            print(goodbye_message)
            break

        elif query.lower() == 'update info':
            update_user_info(preferred_language)
            continue
        
        # Translate user input to English
        translated_query = translate_text(query, source_lang=preferred_language, target_lang="en-IN", output_script="fully-native")
        if not translated_query:
            translation_failed_message = translate_text("Chatbot: Translation failed. Falling back to English...", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or "Chatbot: Translation failed. Falling back to English..."
            print(translation_failed_message)
            translated_query = query  # Fallback to original query if translation fails
        
        # Get top relevant documents from both datasets
        top_docs = find_most_relevant_documents(translated_query)
        
        # Combine into context
        context = "\n\n".join(top_docs)
        
        # Generate an answer from GPT-4o
        answer = generate_answer(context, translated_query)
        
        # Translate the answer back to the user's preferred language
        translated_answer = translate_text(answer, source_lang="en-IN", target_lang=preferred_language, output_script="fully-native")
        if not translated_answer:
            translated_answer = answer  # Fallback to English if translation fails
        
        chatbot_response_prompt = translate_text("Chatbot: ", source_lang="en-IN", target_lang=preferred_language, output_script="fully-native") or "Chatbot: "
        print(f"\n{chatbot_response_prompt}{translated_answer}\n")

        # Log the chat interaction
        log_chat_interaction(
            user_id=user_info.get('user_id', 'anonymous'),  # Use user ID if available, else 'anonymous'
            query=query,
            response=translated_answer,
            language=preferred_language
        )

# ===== RUN THE CHATBOT =====
if __name__ == "__main__":
    chat()