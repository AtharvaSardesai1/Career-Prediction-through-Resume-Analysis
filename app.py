import bcrypt
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
import re
import fitz  # PyMuPDF for handling PDFs
import re
import pickle
import nltk
import language_tool_python
from werkzeug.utils import secure_filename
from Courses import course_mapping
from skills import skills_mapping 
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import sqlite3

app = Flask(__name__)

app.secret_key = 'Guitar'

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# SQLite database path
sqlite_db_path = "sakilla.db"

# Initialize SQLite connection
def get_db_connection():
    connection = sqlite3.connect(sqlite_db_path)
    connection.row_factory = sqlite3.Row  # Allows access to columns by name
    return connection

# Initialize SQLite database and create tables
def initialize_database():
    connection = get_db_connection()
    cursor = connection.cursor()

    # Example users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    connection.commit()
    connection.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor()

        # Fetch the account
        cursor.execute('SELECT * FROM accounts WHERE username = ?', (username,))
        account = cursor.fetchone()
        connection.close()

        if account:
            hashed_password = account['password']
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                # Passwords match, log in the user
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                session['du'] = 'Arnav Prasad'
                msg = 'Logged in successfully!'
                return render_template('user_venue_booking_page.html', msg=msg)
            else:
                msg = 'Incorrect username / password!'
        else:
            msg = 'Incorrect username / password!'

    return render_template('login2.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        connection = get_db_connection()
        cursor = connection.cursor()

        # Check if the account already exists
        cursor.execute('SELECT * FROM accounts WHERE username = ?', (username,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert the account into the database
            cursor.execute('INSERT INTO accounts (username, password, email) VALUES (?, ?, ?)', (username, hashed_password, email))
            connection.commit()
            msg = 'You have successfully registered!'

        connection.close()
    elif request.method == 'POST':
        msg = 'Please fill out the form!'

    return render_template('register2.html', msg=msg)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'admin_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        admin_password = request.form['admin_password']

        connection = get_db_connection()
        cursor = connection.cursor()

        # Fetch account for the given username
        cursor.execute('SELECT * FROM accounts WHERE username = ?', (username,))
        account = cursor.fetchone()
        connection.close()

        if account:
            hashed_password = account['password']
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                # Passwords match, now check admin password
                if admin_password == 'hackathon':  # Admin password is hardcoded
                    # Admin login successful
                    session['admin_loggedin'] = True
                    session['id'] = account['id']
                    session['username'] = account['username']
                    msg = 'Admin logged in successfully!'
                    return render_template('admin_dashboard.html', msg=msg)
                else:
                    msg = 'Incorrect admin password!'
            else:
                # User password incorrect
                msg = 'Incorrect username / password!'
        else:
            # No account found for the given username
            msg = 'Incorrect username / password!'

    return render_template('admin_login.html', msg=msg)


@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'admin_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        admin_password = request.form['admin_password']

        connection = get_db_connection()
        cursor = connection.cursor()

        # Check if the account already exists
        cursor.execute('SELECT * FROM accounts WHERE username = ?', (username,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email or not admin_password:
            msg = 'Please fill out the form!'
        elif admin_password != 'hackathon':  # Admin password is hardcoded
            msg = 'Incorrect admin password!'
        else:
            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert the admin account into the database
            cursor.execute('INSERT INTO accounts (username, password, email) VALUES (?, ?, ?)', (username, hashed_password, email))
            connection.commit()
            msg = 'Admin registered successfully!'
        
        connection.close()

    elif request.method == 'POST':
        msg = 'Please fill out the form!'

    return render_template('admin_register.html', msg=msg)


#ending of admin login and register functions.

# @app.route('/')
# def index():
#     return render_template('cards2.html')


@app.route('/display_admin_dashboard')
def display_admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/workflow')
def workflow():
    return render_template('timeline.html')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
tool = language_tool_python.LanguageTool('en-US')  # Grammar checker
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model_save_path = 'salary_pred_model.pkl'
loaded_model = joblib.load(model_save_path)
nlp = spacy.load("en_core_web_sm")
# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_resume(resume_text):
    # Cleaning function
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text



def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text


skill_categories = {
    "Technical": [
        "Python", "SQL", "Java", "C++", "Machine Learning", "Data Analysis", "Big Data",
        "TensorFlow", "PyTorch", "Cloud Computing", "AWS", "Azure", "GCP", "JavaScript",
        "React", "Angular", "Docker", "Kubernetes", "Linux", "Git", "Tableau", "Power BI",
        "Cybersecurity", "Database Management", "Networking"
    ],
    "Managerial": [
        "Project Management", "Agile", "Scrum", "Budgeting", "Leadership", "Risk Management",
        "Strategic Planning", "Team Management", "Decision Making", "Time Management",
        "Conflict Resolution", "Stakeholder Management", "Resource Allocation", "Negotiation",
        "Goal Setting"
    ],
    "Interpersonal": [
        "Communication", "Problem Solving", "Creativity", "Adaptability", "Collaboration",
        "Empathy", "Public Speaking", "Presentation Skills", "Listening", "Networking",
        "Teamwork", "Emotional Intelligence", "Conflict Resolution", "Critical Thinking"
    ]
}


def extract_skills(resume_text):
    # Initialize lists for categorized skills
    technical_skills = []
    managerial_skills = []
    interpersonal_skills = []

    # Use NLP processing for tokenization
    doc = nlp(resume_text)

    # Extract and categorize skills
    for token in doc:
        skill = token.text
        if skill in skill_categories["Technical"] and skill not in technical_skills:
            technical_skills.append(skill)
        elif skill in skill_categories["Managerial"] and skill not in managerial_skills:
            managerial_skills.append(skill)
        elif skill in skill_categories["Interpersonal"] and skill not in interpersonal_skills:
            interpersonal_skills.append(skill)

    return technical_skills, managerial_skills, interpersonal_skills

def load_and_prepare_data(dataset_file):
    # Load dataset from CSV file
    dataset = pd.read_csv(dataset_file)

    # Split dataset into train, validation, and test sets (80-10-10 split)
    train_df, temp_df = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Convert DataFrames to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Combine into a single DatasetDict
    dataset_dict = {'train': train_dataset, 'validation': valid_dataset, 'test': test_dataset}
    
    # Initialize tokenizer for DistilBERT
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenize dataset
    def preprocess_data(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    # Apply preprocessing
    encoded_dataset = {split: data.map(preprocess_data, batched=True) for split, data in dataset_dict.items()}
    
    # Label conversion for Trainer (labels must be integers)
    label_mapping = {label: i for i, label in enumerate(set(train_df['label']))}
    for split in encoded_dataset:
        encoded_dataset[split] = encoded_dataset[split].map(lambda x: {'label': label_mapping[x['label']]})
    
    return encoded_dataset, label_mapping

def load_and_predict(text, label_mapping, model_path="custom-tone-model"):
    # Initialize pipeline with the trained model
    tone_classifier = pipeline("text-classification", model=model_path, tokenizer="distilbert-base-uncased")
    
    # Split text into chunks of 512 tokens or less
    max_length = 512
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")['input_ids']
    
    # Calculate the number of chunks
    num_chunks = (inputs.shape[1] + max_length - 1) // max_length
    chunks = [text[i * max_length : (i + 1) * max_length] for i in range(num_chunks)]
    
    # Predict tone for each chunk
    predictions = [tone_classifier(chunk)[0] for chunk in chunks]
    
    # Extract labels and scores
    labels = [pred['label'] for pred in predictions]
    scores = [pred['score'] for pred in predictions]

    # Find the most common label across chunks, average the confidence scores
    most_common_label = max(set(labels), key=labels.count)
    average_confidence = np.mean(scores)
    
    # Map the label back to its original name
    predicted_label = {v: k for k, v in label_mapping.items()}.get(int(most_common_label.replace("LABEL_", "")), "Unknown")
    
    return predicted_label, average_confidence

def predict_tone(text):
    # Validate input
    print("!!!!!!!!!!!!!!!!!!!!!!")
    if not isinstance(text, str) or not text.strip():
        return "Invalid input: Expected a non-empty string", 0.0
    # Path to your single CSV file
    dataset_file = 'tone_dataset.csv'
    print("222222222222222222222222222222222")
    # Load and prepare data
    encoded_dataset, label_mapping = load_and_prepare_data(dataset_file)
    print(label_mapping)
    print("333333333333333333333333333333")
    predicted_label, confidence = load_and_predict(text, label_mapping)
    print("4444444444444444444444444444")
    return predicted_label,confidence

# gnn starts----------------------------------------------------------
class Graph: 
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
        self.nodes = [
            # Base nodes -13
            "Web Developer", "Data Scientist", "DevOps Engineer",
            "Java Developer", "Testing", "Python Developer", "Blockchain Developer",
            "ETL Developer", "Database Engineer", ".NET Developer",
            "Automation Testing Engineer", "Network Security Engineer", "SAP Developer",

            # Non-shared future nodes -33-1-1
            "UI/UX Designer", "API Developer",
            "Software Architect", "Predictive Analytics Specialist", "Data Product Manager", "Chief Data Officer",
            "AI Research Scientist", "Platform Engineer", "Cloud Database Administrator", "Database Performance Engineer",
            "NoSQL Specialist", "Ethical Hacker", "Cloud Security Specialist",
            "Incident Response Manager", "SAP Functional Consultant", "SAP Integration Specialist", "ERP Architect", 
            "SAP S/4HANA Specialist", "Blockchain Architect", "Smart Contract Developer", "Cryptography Expert", "DApp Developer",
            "Azure Cloud Specialist", "Full Stack .NET Developer", "Cloud Data Integration Specialist", "Big Data Engineer", "Data Warehouse Architect",
            "Test Automation Architect", "Continuous Integration Engineer", "Performance Optimization Specialist","Bioinformatics Developer",
            
            
            # Shared future nodes -10
            "Data Analyst", "Android Developer", "Data Architect", "Full-Stack Developer", "AI/ML Engineer", "Quality Assurance Lead", 
            "Cloud Architect", "Cybersecurity Architect", "Security Automation Specialist","Site Reliability Engineer"
        ]
        print(f"Length of self.nodes: {len(self.nodes)}")  # Should be 54
        print(f"Unique nodes: {len(set(self.nodes))}")  # Should also be 54

        self.edges = [
            # Base to non-shared 
            ("Web Developer", "UI/UX Designer"), ("Web Developer", "API Developer"),("Java Developer", "Software Architect"),
            ("Data Scientist", "Predictive Analytics Specialist"), ("Data Scientist", "Data Product Manager"),("Data Scientist", "AI Research Scientist"), ("Data Scientist", "Chief Data Officer"),
            ("Database Engineer", "Cloud Database Administrator"), ("Database Engineer", "Database Performance Engineer"),("Database Engineer", "NoSQL Specialist"),    
            ("DevOps Engineer", "Platform Engineer"), 
            ("Network Security Engineer", "Ethical Hacker"), ("Network Security Engineer", "Incident Response Manager"), ("Network Security Engineer", "Cloud Security Specialist"),
            ("SAP Developer", "SAP Functional Consultant"), ("SAP Developer", "SAP S/4HANA Specialist"), ("SAP Developer", "SAP Integration Specialist"), ("SAP Developer", "ERP Architect"),
            ("Blockchain Developer", "Smart Contract Developer"), ("Blockchain Developer", "Cryptography Expert"), ("Blockchain Developer", "DApp Developer"), ("Blockchain Developer", "Blockchain Architect"),
            (".NET Developer", "Azure Cloud Specialist"),  (".NET Developer", "Full Stack .NET Developer"),
            ("ETL Developer", "Data Warehouse Architect"), ("ETL Developer", "Cloud Data Integration Specialist"), ("ETL Developer", "Big Data Engineer"),
            (".NET Developer", "Azure Cloud Specialist"), ("Automation Testing Engineer", "Test Automation Architect"),
            ("Automation Testing Engineer", "Test Automation Architect") , ("Automation Testing Engineer", "Continuous Integration Engineer"), ("Automation Testing Engineer", "Performance Optimization Specialist"),
            ("Python Developer", "Bioinformatics Developer"),
            
            # Base to shared 
            ("Web Developer", "Full-Stack Developer"), ("Java Developer", "Full-Stack Developer"),("Python Developer", "Full-Stack Developer"),
            ("Python Developer", "Data Analyst"), ("Data Scientist", "Data Analyst"),
            ("Web Developer", "Android Developer"), ("Java Developer", "Android Developer"),
            ("Python Developer", "AI/ML Engineer"), ("Automation Testing Engineer", "AI/ML Engineer"),("Data Scientist", "AI/ML Engineer"),
            ("Testing", "Quality Assurance Lead"),("Automation Testing Engineer", "Quality Assurance Lead"),("DevOps Engineer", "Quality Assurance Lead"),
            ("DevOps Engineer", "Site Reliability Engineer"), ("Network Security Engineer", "Cybersecurity Architect"),
            ("SAP Developer", "Data Architect"), ("Database Engineer", "Data Architect"), ("Data Scientist", "Data Architect"),
            ("ETL Developer", "Cloud Architect"),("DevOps Engineer", "Cloud Architect"),("Database Engineer", "Cloud Architect"),
            ("DevOps Engineer", "Security Automation Specialist"), ("Network Security Engineer", "Security Automation Specialist"),
            ("Network Security Engineer", "Cybersecurity Architect"), ("SAP Developer", "Cybersecurity Architect"),
            ("Network Security Engineer", "Site Reliability Engineer"), ("Python Developer", "Site Reliability Engineer"), ("DevOps Engineer", "Cybersecurity Architect"),

            # Base to base
            ("Python Developer", "Data Scientist"), 
            ("Testing", "Automation Testing Engineer"),
            ("Testing", "DevOps Engineer"), (".NET Developer", "DevOps Engineer"), ("Java Developer", "DevOps Engineer")
        ]
        
        edge_nodes = set(node for edge in self.edges for node in edge)
        extra_edge_nodes = edge_nodes - set(self.nodes)
        print(f"Nodes in edges not in self.nodes: {extra_edge_nodes}")
        
    def create_graph(self) :
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
        return self.graph

# graph_instance = Graph()

# G = graph_instance.create_graph()

career_nodes = {
    "Web Developer": [1, 0, 1, 0.6, 1],  # Python-based, Java-based, Web-based, Salary, Experience
    "Data Scientist": [1, 0, 0, 0.8, 2],
    "DevOps Engineer": [1, 1, 0, 0.7, 2],
    "Java Developer": [0, 1, 0, 0.6, 1],
    "Testing": [1, 1, 0, 0.3, 0],
    "Python Developer": [1, 0, 0, 0.7, 1],
    "Blockchain Developer": [1, 0, 0, 0.9, 2],
    "ETL Developer": [1, 0, 0, 0.6, 1],
    "Database Engineer": [1, 0, 0, 0.6, 2],
    ".NET Developer": [0, 1, 1, 0.5, 1],
    "Automation Testing Engineer": [1, 0, 0, 0.4, 1],
    "Network Security Engineer": [1, 1, 0, 0.8, 2],
    "SAP Developer": [0, 1, 0, 0.7, 1],
    "UI/UX Designer": [0, 0, 1, 0.5, 1],
    "API Developer": [1, 0, 1, 0.7, 2],
    "Software Architect": [0, 1, 1, 1.0, 2],
    "Predictive Analytics Specialist": [1, 0, 0, 0.8, 2],
    "Data Product Manager": [1, 0, 0, 0.9, 2],
    "Chief Data Officer": [1, 0, 0, 1.0, 2],
    "AI Research Scientist": [1, 0, 0, 0.9, 2],
    "Platform Engineer": [1, 1, 0, 0.8, 2],
    "Cloud Database Administrator": [1, 0, 0, 0.8, 1],
    "Database Performance Engineer": [1, 0, 0, 0.7, 1],
    "NoSQL Specialist": [1, 0, 0, 0.7, 1],
    "Cybersecurity Architect": [0, 1, 0, 0.9, 2],
    "Ethical Hacker": [1, 0, 0, 0.8, 1],
    "Cloud Security Specialist": [1, 0, 1, 0.7, 1],
    "Incident Response Manager": [1, 0, 0, 0.7, 1],
    "SAP Functional Consultant": [0, 1, 0, 0.6, 1],
    "SAP Integration Specialist": [0, 1, 0, 0.7, 1],
    "ERP Architect": [1, 1, 0, 0.8, 2],
    "SAP S/4HANA Specialist": [0, 1, 0, 0.9, 2],
    "Blockchain Architect": [1, 0, 0, 0.9, 2],
    "Smart Contract Developer": [1, 0, 0, 0.9, 2],
    "Cryptography Expert": [1, 0, 0, 0.9, 2],
    "DApp Developer": [1, 0, 0, 0.9, 2],
    "Azure Cloud Specialist": [1, 1, 0, 0.8, 2],
    "Full Stack .NET Developer": [0, 1, 1, 0.7, 1],
    "Cloud Data Integration Specialist": [1, 0, 0, 0.7, 2],
    "Big Data Engineer": [1, 0, 0, 0.8, 2],
    "Data Warehouse Architect": [1, 0, 0, 0.9, 2],
    "Test Automation Architect": [1, 0, 0, 0.8, 2],
    "Continuous Integration Engineer": [1, 1, 0, 0.8, 1],
    "Performance Optimization Specialist": [1, 0, 0, 0.7, 2],
    "Bioinformatics Developer": [1, 0, 0, 0.7, 2],
    "Data Analyst": [1, 0, 0, 0.8, 2],
    "Android Developer": [1, 0, 1, 0.6, 1],
    "Data Architect": [1, 0, 0, 0.8, 2],
    "Full-Stack Developer": [1, 0, 1, 0.8, 2],
    "AI/ML Engineer": [1, 0, 0, 0.9, 2],
    "Quality Assurance Lead": [1, 0, 0, 0.7, 1],
    "Cloud Architect": [1, 1, 0, 0.9, 2],
    "Security Automation Specialist": [0, 1, 0, 0.8, 1],
    "Site Reliability Engineer": [1, 1, 0, 0.9, 2]
}

# Convert the dictionary into a tensor
# career_tensor = torch.tensor(list(career_nodes.values()))

# # Display the tensor
# career_tensor

# Step 3: GCN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        # First layer (Graph Convolution)
        x = F.relu(self.conv1(x, edge_index))
        # Second layer (Graph Convolution)
        x = self.conv2(x, edge_index)
        return x
# print(G.nodes)
# print(f"Number of nodes: {len(G.nodes)}")
# print(f"Number of edges: {len(G.edges)}")



def prepare_data(G, career_tensor):
    node_mapping = {node: idx for idx, node in enumerate(G.nodes)}
    
    valid_edges = [(u, v) for u, v in G.edges if u in node_mapping and v in node_mapping]

    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in valid_edges], dtype=torch.long).t().contiguous()

    # Create a Data object from node features and edge_index
    data = Data(x=career_tensor, edge_index=edge_index)
    
    return data

# Prepare the data
# data = prepare_data(G, career_tensor)

# Display information about the created data
# print(data)


# Initialize the GCN model
# model = GCN(in_channels=career_tensor.shape[1], out_channels=2)  # Example output features = 2

# # Set optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 6: Train the GCN model (dummy training loop)
def train(model, data, optimizer, epochs=200):
    for epoch in range(epochs):
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index)
        # print(f"Shape of out: {out.shape}")
        # print(f"Shape of data.x: {data.x.shape}")

        # Loss (mean squared error for this example)
        loss = F.mse_loss(out, data.x[:, :2])

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Start the training process
# train(model, data, optimizer)

# # Step 7: Visualize the output (optional)
# model.eval()
# out = model(data.x, data.edge_index).detach()

# # For visualization, take the first 2 features
# plt.scatter(out[:, 0].numpy(), out[:, 1].numpy())
# plt.title('Node Embeddings after GCN')
# plt.show()

def suggest_career_paths(graph, job_role, career_nodes, edges, max_paths=8):
    """
    Suggest career paths based on the given job role and connected nodes.

    Args:
        job_role (str): The input job role.
        career_nodes (dict): Dictionary containing job roles and their attributes.
            Each role has attributes [Python-based, Java-based, Web-based, Salary, Experience].
        edges (list of tuples): List of tuples representing connections between job roles.
        max_paths (int): Maximum number of career paths to suggest.

    Returns:
        list: Suggested career paths with required programming languages/domains.
    """
    if job_role not in career_nodes:
        print(f"Error: '{job_role}' not found in the career nodes.")
        return []

    # Extract the attributes of the given job role
    role_attributes = career_nodes[job_role]
    required_salary = role_attributes[3]
    required_experience = role_attributes[4]

    # Filter roles based on criteria
    suggested_roles = []
    for role, attributes in career_nodes.items():
        if (
            attributes[3] >= required_salary and  # Salary criteria
            attributes[4] >= required_experience and  # Experience criteria
            role != job_role  and # Exclude the current job role
            graph.has_edge(job_role, role)  # Check if the role is connected to the job_role in the graph

        ):
            suggested_roles.append((role, attributes))
    print(suggested_roles)
    # Sort the roles by salary and experience for better suggestions
    suggested_roles.sort(key=lambda x: (x[1][3], x[1][4]), reverse=True)

    # If fewer than max_paths, predict from connected nodes
    print(f"Length of suggested_roles: {len(suggested_roles)}")
    if len(suggested_roles) < max_paths:
        print(f"Checking neighbors of {job_role}: {list(graph.neighbors(job_role))}")
        for connected_role in graph.neighbors(job_role):
            # print(f"Evaluating connected role: {connected_role}")
            # Check if connected role meets the salary and experience criteria
            if connected_role != job_role:  # Skip the current job role
                connected_role_data = career_nodes.get(connected_role)
                if connected_role_data is not None:
                    connected_role_salary = connected_role_data[3]
                    connected_role_experience = connected_role_data[4]
                    # Check if the neighbor's salary and experience meet the criteria
                    if connected_role_salary >= required_salary and connected_role_experience >= required_experience:
                        print(f"Adding {connected_role} based on salary and experience criteria")
                        # Check if it's already in the suggested roles list
                        if not any(r[0] == connected_role for r in suggested_roles):
                            suggested_roles.append((connected_role, connected_role_data))
                        else:
                            print(f"Role {connected_role} is already in suggested roles.")
                else:
                    print(f"No data found for {connected_role}, skipping.")

        # Now check the neighbors of the neighbors if still fewer than max_paths
        if len(suggested_roles) < max_paths:
            print(f"Checking neighbors of the neighbors of {job_role}")
            for connected_role in graph.neighbors(job_role):
                # Now check the neighbors of the connected_role
                for connected_neighbor in graph.neighbors(connected_role):
                    # Check if connected_neighbor meets the salary and experience criteria
                    connected_neighbor_data = career_nodes.get(connected_neighbor)
                    if connected_neighbor_data is not None:
                        connected_neighbor_salary = connected_neighbor_data[3]
                        connected_neighbor_experience = connected_neighbor_data[4]
                        # Only add if they meet the salary and experience requirements
                        if connected_neighbor_salary >= required_salary and connected_neighbor_experience >= required_experience:
                            print(f"Adding {connected_neighbor} from neighbor's neighbors")
                            # Check if it's already in the suggested roles list
                            if not any(r[0] == connected_neighbor for r in suggested_roles):
                                suggested_roles.append((connected_neighbor, connected_neighbor_data))
                            else:
                                print(f"Role {connected_neighbor} is already in suggested roles.")


        print(suggested_roles)
        # Limit the suggestions to max_paths
        suggested_roles = suggested_roles[:max_paths]

    # Format the output with required languages/domains
    result = []
    for role, attributes in suggested_roles:
        languages = []
        if attributes[0]:
            languages.append("Python")
        if attributes[1]:
            languages.append("Java")
        if attributes[2]:
            languages.append("Web")

        result.append({
            "Role": role,
            "Salary": attributes[3],
            "Experience": attributes[4],
            "Required Languages": languages
        })

    return result


@app.route('/upload', methods=['POST'])
def upload():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    email = request.form['email']
    education = request.form['education']
    experience = request.form['experience']
    location = request.form['location']
    gender = request.form['gender']
    age = request.form['age']
    if 'resume_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    resume_file = request.files['resume_file']
    if resume_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filename = secure_filename(resume_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(file_path)

    # Extract resume text based on file type
    if filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(open(file_path, "rb"))
    elif filename.endswith('.txt'):
        resume_text = open(file_path, "r", encoding="utf-8").read()
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Clean and analyze the resume
    cleaned_resume = clean_resume(resume_text)
    tech_skills, mgr_skills, inter_skills = extract_skills(resume_text)

    # grammar_errors = grammar_check(resume_text)
    
    dataset_file = 'tone_dataset.csv'
    
    # Load and prepare data
    encoded_dataset, label_mapping = load_and_prepare_data(dataset_file)
    print(label_mapping)
    
    predicted_tone, predicted_confidence = predict_tone(resume_text)
    # predicted_tone, confidence = load_and_predict(resume_text, label_mapping)
    # print(predicted_tone)
    # print(confidence)
    # Use your existing model to predict the category
    input_features = tfidfd.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]

    # Map prediction to category
    category_mapping = {
        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
        20: "Python Developer", 24: "Web Developer", 12: "HR",
        13: "Hadoop", 3: "Blockchain Developer", 10: "ETL Developer",
        18: "Operations Manager", 6: "Data Scientist", 22: "Sales",
        16: "Mechanical Engineer", 1: "Arts", 7: "Database Engineer",
        11: "Electrical Engineering", 14: "Health and Fitness",
        19: "PMO", 4: "Business Analyst", 9: ".NET Developer",
        2: "Automation Testing Engineer", 17: "Network Security Engineer",
        21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
    }
    category_name = category_mapping.get(prediction_id, "Unknown")
    recommended_skills = skills_mapping.get(category_name, [])
    recommended_courses = course_mapping.get(category_name, [])

    
    if education == "Bachelor's":
        new_candidate = pd.DataFrame({
    'Education Level': ['Bachelor\'s'],
    'Years of Experience': [experience],
    'Job Title': [category_name],
    'Age': [age],
    'Gender': [gender]
    })
    elif education == "Master's":
        new_candidate = pd.DataFrame({
    'Education Level': ['Master\'s'],
    'Years of Experience': [experience],
    'Job Title': [category_name],
    'Age': [age],
    'Gender': [gender]
    })
    else :
        new_candidate = pd.DataFrame({
    'Education Level': ['PhD'],
    'Years of Experience': [experience],
    'Job Title': [category_name],
    'Age': [age],
    'Gender': [gender]
    })
    # new_candidate = pd.DataFrame({
    # 'Education': education_value,
    # 'Experience': [experience],
    # 'Location': [location],
    # 'Job_Title': [category_name],
    # 'Age': [age],
    # 'Gender': [gender]
    # })
    predicted_salary = loaded_model.predict(new_candidate)
    salary = round(predicted_salary[0])
    
    graph_instance = Graph()

    G = graph_instance.create_graph()
    career_tensor = torch.tensor(list(career_nodes.values()))

# Display the tensor
    career_tensor
    data = prepare_data(G, career_tensor)
    model = GCN(in_channels=career_tensor.shape[1], out_channels=2)  # Example output features = 2

# Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, data, optimizer)

# Step 7: Visualize the output (optional)
    model.eval()
    out = model(data.x, data.edge_index).detach()
    job_role = category_name
    career_paths = suggest_career_paths(G,job_role, career_nodes, G.edges)
    print(f"Suggested career paths for '{job_role}':")
    for path in career_paths:
        print(f"Role: {path['Role']}, Salary: {path['Salary']}, Experience: {path['Experience']}, Required Languages: {', '.join(path['Required Languages'])}")

    
    
    print("First Name:", first_name)
    print("Last Name:", last_name)
    print("Email:", email)
    print("Age:", age)
    # print(salary)
    print(category_name)
    print(recommended_courses)
    print(recommended_skills)
    return render_template('data_display.html', fname = first_name, lname = last_name,tech_skills=tech_skills, mgr_skills=mgr_skills, inter_skills=inter_skills ,age=age,location=location,education=education,tone=predicted_tone, domain=category_name,salary=salary, skills=recommended_skills, courses = recommended_courses, career_paths=career_paths)
    
    

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)


print('done!!')