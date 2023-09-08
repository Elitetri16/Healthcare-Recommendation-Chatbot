import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import ttk
import nltk


# Initialize NLTK and download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample healthcare recommendations (generic)
disease_recommendations = {
    "Cold": "Rest, drink fluids, and consider over-the-counter cold medication.",
    "Flu": "Rest, stay hydrated, and consult a doctor if symptoms worsen.",
    "Diabetes": "Monitor blood sugar levels, follow a healthy diet, and take prescribed medications as directed.",
    "COVID-19": "Isolate yourself, get tested, and follow public health guidelines. Seek medical help if symptoms worsen.",
    "Pneumonia": "Consult a doctor immediately, as pneumonia can be a severe condition requiring antibiotics.",
    "Dysentery": "Stay hydrated, rest, and consult a doctor if diarrhea and abdominal pain persist.",
    "Psychological": "For those grappling with psychological issues, seeking professional help is paramount.Psychotherapy, including cognitive-behavioral therapy and mindfulness practicescan offer invaluable support. Medication, when prescribed by a psychiatrist, maycomplement therapy for specific conditions. Lifestyle changes such as regular exercise,proper nutrition, and adequate sleep contribute to mental well-being. A robust supportsystem, including friends and family, plays a vital role. Encouraging self-care, setting achievable goals, and maintaining a journal for self-expression are steps toward recovery.Education about the condition reduces stigma, while regulacheck-ins with mental health professionals ensure progress. Timely intervention and personalizedcare are fundamental in navigating psychological challenges"
}

# Function to provide recommendations based on disease
def get_recommendations(disease):
    return disease_recommendations.get(disease, "Recommendations not available for this disease.")

# Function to preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Function to recommend disease and healthcare solutions
def recommend_disease_and_solutions(patient_details):
    # Sample diseases and their associated symptoms
    diseases = {
        "Cold": ["runny nose", "sneezing", "cough"],
        "Flu": ["fever", "headache", "body aches"],
        "Diabetes": ["frequent urination", "increased thirst", "fatigue"],
        "COVID-19": ["fever", "cough", "shortness of breath","chills,difficulty breathing,fatigue,Muscle or body aches,headache"],
        "Pneumonia": ["cough, which may produce greenish", "yellow or even bloody mucus","fever","sweating","shaking chills","shortness of breath", "Rapid, shallow breathing","sharp or stabbing chest pain  that gets worse when you breathe deeply or cough","loss of appetite","low energy", "fatigue"],
        "Dysentery": ["diarrhea","high fever", "weight loss", "upset stomach", "nausea", "vomiting","bacillary dysentery","diarrhea", "blood", "mucus"],
        "Psychological": ["feeling sad","reduced ability to concentrate","loss of apetite", "stress", "lost friends","anxiety", "depression","excessive fears", "worries","extreme mood swings", "withdrawal from friends and activities","tiredness", "low energy", "detachment from reality","delusions", "paranoia","hallucinations","Inability to cope with daily problems", "trouble understanding and relating to situations and to people","alchohol dependency","suicidal thoughts"]
      }
    # Preprocess patient details
    patient_details = preprocess_text(patient_details)

    # Calculate TF-IDF vectors for diseases and patient details
    tfidf_vectorizer = TfidfVectorizer()
    disease_vectors = tfidf_vectorizer.fit_transform([' '.join(symptoms) for symptoms in diseases.values()])
    patient_vector = tfidf_vectorizer.transform([patient_details])

    # Calculate cosine similarity between patient details and disease symptoms
    similarities = cosine_similarity(patient_vector, disease_vectors)

    # Identify the disease with the highest similarity
    max_similarity_index = similarities.argmax()
    disease = list(diseases.keys())[max_similarity_index]

    # If no disease matches the symptoms
    if similarities.max() < 0.2:
        return None, "None, you're fit!"

    # Get recommendations for the identified disease
    recommendations = get_recommendations(disease)

    return disease, recommendations


# Function to handle the chat interaction
def chat():
    patient_details = input_text.get("1.0", "end-1c")
    disease, recommendations = recommend_disease_and_solutions(patient_details)
    chat_output.config(state=tk.NORMAL)
    chat_output.insert(tk.END, f"Patient Details: {patient_details}\n")
    chat_output.insert(tk.END, f"Identified Disease: {disease}\n")
    chat_output.insert(tk.END, f"Recommendations: {recommendations}\n\n")
    chat_output.config(state=tk.DISABLED)
    input_text.delete("1.0", tk.END)

# Create the main window
root = tk.Tk()
root.title("Healthcare Recommendation Chatbot")

# Set background color and add a healthcare logo image
background_color = "blue"  # Blue background color
root.configure(bg=background_color)

# Load and display a healthcare logo
logo_image = Image.open("healthcare_logo.png")  # logo image file path
logo_image = logo_image.resize((100, 100))  # Resize the logo as needed
logo_image = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo_image, bg=background_color)
logo_label.pack(pady=10)

# Chat UI
chat_frame = ttk.LabelFrame(root, text="Chatbox")
chat_frame.pack(padx=10, pady=10, fill="both", expand="yes")

# Set styles for chat input and output
style = ttk.Style()
style.configure("TButton", padding=(10, 5), font=('Helvetica', 12))
style.configure("TLabel", font=('Helvetica', 12))
style.configure("TText", font=('Helvetica', 12),)

input_text = tk.Text(chat_frame, wrap=tk.WORD, width=40, height=10, bg="white")
input_text.pack(pady=5)

chat_button = ttk.Button(chat_frame, text="Recommend", command=chat)
chat_button.pack(pady=5)

chat_output = tk.Text(chat_frame, wrap=tk.WORD, width=40, height=10, state=tk.DISABLED, bg="white")
chat_output.pack(pady=5)

# Run the GUI
root.mainloop()

