import os
import mysql.connector
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session
from flask_session import Session
from datetime import datetime, timedelta
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import lead_processing
from llm_service import get_llm
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from dotenv import load_dotenv
import feedparser
from werkzeug.utils import secure_filename
import re
import uuid
import json

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'
Session(app)

# Configure Flask secret key
if not os.getenv("FLASK_SECRET_KEY"):
    logger.error("FLASK_SECRET_KEY not found in environment variables.")
    raise ValueError("FLASK_SECRET_KEY not found in environment variables.")
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify lead_processing module
try:
    logger.info("Attempting to access lead_processing.process_leads")
    getattr(lead_processing, 'process_leads')
    logger.info("Successfully accessed lead_processing.process_leads")
except AttributeError as e:
    logger.error("Failed to access lead_processing.process_leads: %s", e)
    raise
    
user_chains = {}

MYSQL_CONFIG = {
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'host': os.getenv("MYSQL_HOST"),
    'database': os.getenv("MYSQL_CHAT_DB"),
    'raise_on_warnings': True
}

db_config = {
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'host': os.getenv("MYSQL_HOST"),
    'database': os.getenv("MYSQL_CONTENT_DB"),
    'raise_on_warnings': True
}

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
    raise ValueError("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
COLLECTION_NAME = "digidara_website_info"

# Email Setup for SMTP
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    logger.error("EMAIL_ADDRESS or EMAIL_PASSWORD not found in environment variables.")
    raise ValueError("EMAIL_ADDRESS or EMAIL_PASSWORD not found in environment variables.")
MAIL_SERVER = 'smtpout.secureserver.net'



MAIL_PORT = 587  # Use 587 for TLS

# Medium Feed URL
MEDIUM_FEED = "https://medium.com/feed/@senthilsprd"

# Configure file upload settings
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Sentence Transformer for embeddings (lazy-loaded)
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return embedder

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_medium_posts():
    try:
        feed = feedparser.parse(MEDIUM_FEED)
        if feed.bozo:
            logger.error(f"Error parsing Medium feed: {feed.bozo_exception}")
            return []
        posts = []
        for entry in feed.entries:
            post = {
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "summary": entry.summary
            }
            posts.append(post)
        return posts
    except Exception as e:
        logger.error(f"Error fetching Medium posts: {e}")
        return []

posts = []  # List to store submitted data

# Send contact form email
def send_contact_email(name, email, phone, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg['Subject'] = "New Contact Form Submission"

    body = f"""
    <h2 style="color: #2E7D32; font-family: Arial, sans-serif;">New Contact Form Submission</h2>
    <p style="font-family: Arial, sans-serif;">A new message has been submitted via the DigiDara Technologies contact form:</p>
    <ul style="color: #424242; font-family: Arial, sans-serif;">
        <li><strong>Name:</strong> {name}</li>
        <li><strong>Email:</strong> {email}</li>
        <li><strong>Phone:</strong> {phone}</li>
        <li><strong>Message:</strong> {message}</li>
    </ul>
    <p style="font-family: Arial, sans-serif;">Please follow up with the sender as needed.</p>
    <p style="font-family: Arial, sans-serif;">Best,<br>DigiDARA Team</p>
    """
    html_part = MIMEText(body, 'html')
    msg.attach(html_part)

    try:
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Contact form email sent to %s", EMAIL_ADDRESS)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check your email and password.")
        return False
    except Exception as e:
        logger.error("Error sending contact form email: %s", e)
        return False
    

def send_contact_email(name, profession, interest, email, phone, date, time, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg['Subject'] = "New Contact Form Submission"

    body = f"""
    <h2 style="color: #2E7D32; font-family: Arial, sans-serif;">New Contact Form Submission</h2>
    <p style="font-family: Arial, sans-serif;">A new message has been submitted via the DigiDara Technologies contact form:</p>
    <ul style="color: #424242; font-family: Arial, sans-serif;">
        <li><strong>Name:</strong> {name}</li>
        <li><strong>Profession:</strong> {profession}</li>
        <li><strong>Interest:</strong> {interest}</li>
        <li><strong>Email:</strong> {email}</li>
        <li><strong>Phone:</strong> {phone}</li>
        <li><strong>Date:</strong> {date}</li>
        <li><strong>Time:</strong> {time}</li>
        <li><strong>Message:</strong> {message}</li>
    </ul>
    <estis/dtdara Team</p>
    """
    html_part = MIMEText(body, 'html')
    msg.attach(html_part)

    try:
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Contact form email sent to %s", EMAIL_ADDRESS)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check your email and password.")
        return False
    except Exception as e:
        logger.error("Error sending contact form email: %s", e)
        return False

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    data = request.get_json()
    name = data.get('name')
    profession = data.get('profession')
    interest = data.get('interest')
    email = data.get('email')
    phone = data.get('phone')
    date = data.get('date')
    time = data.get('time')
    message = data.get('message')

    if not all([name, profession, interest, email, phone, date, time]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    success = send_contact_email(name, profession, interest, email, phone, date, time, message)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Failed to send email"}), 500

@app.route('/success')
def success():
    return "Form submitted successfully!"

# Initialize MySQL Database for Chatbot
def init_db():
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                phone VARCHAR(20) NOT NULL,
                course VARCHAR(255) NOT NULL,
                course_duration VARCHAR(50) NOT NULL,
                message TEXT,
                date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute("SHOW COLUMNS FROM users LIKE 'date'")
        result = cursor.fetchone()
        if not result:
            cursor.execute('''
                ALTER TABLE users
                ADD COLUMN date DATETIME DEFAULT CURRENT_TIMESTAMP
            ''')
            logger.info("Added 'date' column to users table.")
        conn.commit()
        logger.info("MySQL database initialized successfully.")
    except mysql.connector.Error as err:
        if err.errno == 1050:
            logger.info("Table 'users' already exists, skipping creation.")
        else:
            logger.error(f"Error initializing MySQL database: {err}")
            raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# Check for Existing User by Email and Phone
def check_existing_user(email, phone):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, course, course_duration FROM users WHERE email = %s AND phone = %s LIMIT 1
        ''', (email, phone))
        result = cursor.fetchone()
        if result:
            return {'name': result[0], 'course': result[1], 'course_duration': result[2]}
        return None
    except mysql.connector.Error as err:
        logger.error(f"Error checking existing user: {err}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# Save or Update User Data in MySQL
def save_user(name, email, phone, course, course_duration, message='', last_processed_time=None, llm=None):
    # Basic validation
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not email or not re.match(email_pattern, email):
        raise ValueError("Invalid email format")
    phone_pattern = r'^\d{10}$'
    if not phone or not re.match(phone_pattern, phone):
        raise ValueError("Phone number must be a 10-digit number")

    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            SELECT message, course, course_duration FROM users WHERE email = %s AND phone = %s LIMIT 1
        ''', (email, phone))
        result = cursor.fetchone()

        if result:
            existing_message = result[0] if result[0] else ''
            existing_course = result[1]
            existing_course_duration = result[2]
            course_to_save = course if course else existing_course
            course_duration_to_save = course_duration if course_duration else existing_course_duration
            new_message = f"{existing_message} | {message}" if existing_message and message else message or existing_message
            cursor.execute('''
                UPDATE users
                SET course = %s, course_duration = %s, message = %s, date = %s
                WHERE email = %s AND phone = %s
            ''', (course_to_save, course_duration_to_save, new_message, current_date, email, phone))
            logger.info(f"Updated user {name} in MySQL database with new message: {new_message}, date: {current_date}")
        else:
            cursor.execute('''
                INSERT INTO users (name, email, phone, course, course_duration, message, date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (name, email, phone, course, course_duration, message, current_date))
            logger.info(f"Saved new user {name} to MySQL database with date {current_date}")

        conn.commit()
        return current_date
    except mysql.connector.Error as err:
        logger.error(f"Error saving/updating user to MySQL: {err}")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# Retrieve relevant context from Qdrant for RAG
def retrieve_from_qdrant(query):
    try:
        embedder = get_embedder()
        query_embedding = embedder.encode(query).tolist()
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3,
            with_payload=True
        )
        if search_result:
            context = " ".join([f"{res.payload['title']}: {res.payload['content']}" for res in search_result])
            context = context.replace("python-ai-foundations.txt Part 5", "")
            return context[:500] if context else "DigiDara offers AI courses and consulting with expert trainers."
        return "DigiDara offers AI courses and consulting with expert trainers."
    except Exception as e:
        logger.error(f"Error retrieving from Qdrant: {e}")
        return "DigiDara offers AI courses and consulting with expert trainers."

# LangChain Setup
llm = get_llm()
mcp_prompt_template = PromptTemplate(
    input_variables=["context", "history", "input"],
    template="""
**System Context (MCP Role Definition):**
You are a persistent agent for DigiDara Technologies, an AI-specialized company, simulating stateful reasoning and strategic planning.
- **Company Info**: Name - DigiDara Technologies, Location - Periya Samy Tower, 2nd Floor, Contact - 9500406945, Timing - 9:30 AM to 7:30 PM.
- **Founder**: Senthil Rajamarthandan, Founder & Managing Director.
- **Courses**: Python, Data Analysis, Data Science, GEN AI, Agentic AI, AI in Cloud Computing, AI digital Marketing, Agentic setups.
- **Protocol Rules**:
  - Response out of context is not allowed.
  - Maintain long-term memory via history.
  - Respond concisely (under 30 words).
  - No price disclosure; if asked, say 'Contact for price' with support@digidaratechnologies.com, 9500406945.
  - For non-DigiDara queries, respond 'Sorry, I can't say.'
  - Strategically promote advantages (e.g., expert trainers, flexible schedules) based on user intent.
- **Context Data**: {context}

**Conversation History (MCP Memory):**
{history}

**User Input (MCP Request):**
{input}

**Assistant Response (MCP Output, under 30 words, with strategic planning):**
"""
)

# Conversation chain with memory
user_chains = {}
init_db()

def get_session_history(user_id):
    if user_id not in user_chains:
        user_chains[user_id] = {
            'history': ChatMessageHistory(),
            'step': 'welcome',
            'questions_asked': 0,
            'name': None,
            'email': None,
            'phone': None,
            'course': None,
            'course_duration': None,
            'last_processed_time': None,
            'last_activity': datetime.now(),
            'has_asked_question': False
        }
    return user_chains[user_id]['history']

chain = (
    RunnablePassthrough.assign(context=lambda x: retrieve_from_qdrant(x["input"])) |
    {"context": lambda x: x["context"], "history": lambda x: x["history"], "input": lambda x: x["input"]} |
    mcp_prompt_template |
    llm
)

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Website Routes
@app.route('/')
def home():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM submissions")
        submissions = cursor.fetchall()
        return render_template('home.html', submissions=submissions)
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to database: {err}")
        flash("Error loading submissions. Please try again later.", "error")
        return render_template('home.html', submissions=[])
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route('/Blog')
def Blog():
    posts = fetch_medium_posts()
    return render_template('more/Blog.html', posts=posts)

@app.route('/consulting')
def consulting():
    return render_template('more/consulting.html')

@app.route('/Contact')
def Contact():
    return render_template('more/Contact.html')

@app.route('/Succss')
def Succss():
    return render_template('more/Succss.html')
@app.route('/nav')
def nav():
    return render_template('nav.html')

@app.route('/Services')
def Services():
    return render_template('Services.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        text = request.form['content']
        link = request.form.get('link', '')

        # Handle uploaded image
        file = request.files.get('image_file')
        filename = ''
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Store filename (not full path) in DB
        query = "INSERT INTO submissions (text, image_link, link) VALUES (%s, %s, %s)"
        values = (text, filename, link)
        cursor.execute(query, values)
        conn.commit()

    cursor.execute("SELECT * FROM submissions")
    submissions = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('form.html', submissions=submissions)

@app.route('/delete_submission/<int:submission_id>', methods=['POST'])
def delete_submission(submission_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Get filename to delete from disk
    cursor.execute("SELECT image_link FROM submissions WHERE id = %s", (submission_id,))
    result = cursor.fetchone()
    if result and result[0]:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], result[0])
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete from database
    cursor.execute("DELETE FROM submissions WHERE id = %s", (submission_id,))
    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('form'))

@app.route('/aichatbot')
def aichatbot():
    return render_template('Services/chatbot.html')

@app.route('/digital')
def digital():
    return render_template('Services/digital.html')

@app.route('/agent')
def agent():
    return render_template('Services/aiproject.html')
@app.route('/masterclass')
def freemasterclass():
    return render_template('freeclass.html')

@app.route('/train')
def train():
    return render_template('Services/lms.html')

@app.route('/course')
def course():
    return render_template('courses/python2.html')

@app.route('/aibeginner')
def aibeginner():
    return render_template('courses/AI_beginner.html')

@app.route('/aibusiness')
def aiforbusiness():
    return render_template('courses/AI_for_business_leaders.html')

@app.route('/aiagency')
def aiforagency():
    return render_template('courses/AI_for_Agency.html')

@app.route('/aicareer')
def aiforcareer():
    return render_template('courses/AI_career.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/AI')
def AI():
    return render_template('AI.html')

@app.route('/event')
def event():
    return render_template('event.html')

# Chatbot Route
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'prompt' not in data:
        logger.error("Invalid request: No prompt provided")
        return jsonify({'response': "Please provide a prompt.", 'quickReplies': None}), 400

    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    prompt = data.get('prompt', '').strip()
    logger.info(f"Received request from {user_id} with prompt: '{prompt}'")

    if user_id not in user_chains:
        user_chains[user_id] = {
            'history': ChatMessageHistory(),
            'step': 'welcome',
            'questions_asked': 0,
            'name': None,
            'email': None,
            'phone': None,
            'course': None,
            'course_duration': None,
            'last_processed_time': None,
            'last_activity': datetime.now(),
            'has_asked_question': False
        }

    user_data = user_chains[user_id]
    state = user_data['step']
    user_data['last_activity'] = datetime.now()

    inactivity_threshold = timedelta(minutes=10)
    if (state in ['free_chat', 'collect_enquiry'] and user_data['has_asked_question'] and
            user_data['last_activity'] < datetime.now() - inactivity_threshold):
        user_data['last_processed_time'] = save_user(
            user_data['name'], user_data['email'], user_data['phone'],
            user_data.get('course'), user_data.get('course_duration'),
            message='Inactive for 10 minutes', last_processed_time=user_data['last_processed_time'], llm=llm
        )
        mcp_context = {
            'last_processed_time': user_data['last_processed_time'],
            'user_id': user_id,
            'action': 'process_leads',
            'intent': 'inactive',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'llm': llm
        }
        try:
            lead_processing.process_leads(mcp_context)
            logger.info(f"Lead processed for {user_data.get('name', 'unknown')} due to inactivity.")
            response = f"Hi {user_data.get('name', '')}! You’ve been inactive for 10 minutes. Lead processed. Email sent. Check Google Sheets."
        except Exception as e:
            logger.error(f"Error processing lead due to inactivity: {e}")
            response = f"Hi {user_data.get('name', '')}! Error processing lead. Check logs. Goodbye!"
        user_chains.pop(user_id, None)
        return jsonify({'response': response, 'quickReplies': None})

    response = ""
    quick_replies = None

    if prompt:
        user_data['questions_asked'] += 1
        if state == 'welcome':
            response = "Great! May I have your name, please?"
            user_data['step'] = 'collect_name'
        elif state == 'collect_name':
            user_data['name'] = prompt
            response = f"Thanks, {user_data['name']}! What’s your email?"
            user_data['step'] = 'collect_email'
        elif state == 'collect_email':
            user_data['email'] = prompt
            response = f"Almost there, {user_data['name']}! What’s your phone number? Please type your 10-digit number."
            user_data['step'] = 'collect_phone'
        elif state == 'collect_phone':
            user_data['phone'] = prompt
            existing_user = check_existing_user(user_data['email'], user_data['phone'])
            if existing_user:
                user_data['name'] = existing_user['name']
                user_data['course'] = existing_user['course']
                user_data['course_duration'] = existing_user['course_duration']
                response = f"Welcome back, {user_data['name']}! How can I help you? (Type 'quit' to exit)"
                user_data['step'] = 'free_chat'
                quick_replies = ['Enquire Now', 'quit']
            else:
                courses = [
                    "Python", "Data Analysis", "Data Science", "GEN AI", "Agentic AI",
                    "AI in Cloud Computing", "AI Digital Marketing", "Agentic Setups"
                ]
                course_list = "\n\t".join(f"{i+1}. {course}" for i, course in enumerate(courses))
                response = f'''Thanks, {user_data['name']}! Which course are you interested in?\n\n We offer:\n{course_list}\nPlease type the course name or number.'''
                user_data['step'] = 'collect_course'
        elif state == 'collect_course':
            user_data['course'] = prompt
            response = f"Great choice, {user_data['name']}! What course duration are you looking for?\n\t1 month\n\t2 months\n\t3 months\n\t6 months"
            user_data['step'] = 'collect_duration'
        elif state == 'collect_duration':
            user_data['course_duration'] = prompt
            user_data['last_processed_time'] = save_user(user_data['name'], user_data['email'], user_data['phone'], user_data['course'], user_data['course_duration'], message='', last_processed_time=user_data['last_processed_time'], llm=llm)
            response = f"Thanks, {user_data['name']}! How can I help? (Type 'quit' to exit)"
            user_data['step'] = 'free_chat'
            quick_replies = ['Enquire Now', 'quit']
        elif state in ['free_chat', 'collect_enquiry']:
            user_data['has_asked_question'] = True
            if prompt.lower() == 'quit':
                if user_data['name'] and user_data['email'] and user_data['phone']:
                    user_data['last_processed_time'] = save_user(
                        user_data['name'], user_data['email'], user_data['phone'],
                        user_data.get('course'), user_data.get('course_duration'),
                        message='',
                        last_processed_time=user_data['last_processed_time'],
                        llm=llm
                    )
                    mcp_context = {
                        'last_processed_time': user_data['last_processed_time'],
                        'user_id': user_id,
                        'action': 'process_leads',
                        'intent': 'quit',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'llm': llm
                    }
                    try:
                        lead_processing.process_leads(mcp_context)
                        logger.info(f"Lead processed for {user_data.get('name', 'unknown')} due to quit command.")
                        response = f"Goodbye, {user_data['name']}! Thank you for contacting us."
                    except Exception as e:
                        logger.error(f"Error processing lead due to quit: {e}")
                        response = f"Goodbye, {user_data['name']}! Error processing lead. Check logs."
                else:
                    response = "Goodbye! Incomplete data, no lead processed."
                user_chains.pop(user_id, None)
                return jsonify({'response': response, 'quickReplies': None})
            elif state == 'free_chat':
                if prompt.lower() == 'enquire now':
                    response = "Please type your enquiry message, and I’ll save it for you. (Type 'quit' to exit)"
                    user_data['step'] = 'collect_enquiry'
                    quick_replies = ['quit']
                else:
                    user_data['last_processed_time'] = save_user(user_data['name'], user_data['email'], user_data['phone'], user_data.get('course'), user_data.get('course_duration'), message=prompt, last_processed_time=user_data['last_processed_time'], llm=llm)
                    try:
                        llm_response = conversational_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": user_id}}
                        )
                        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                        response = response.replace("python-ai-foundations.txt Part 5", "")
                    except Exception as e:
                        logger.error(f"Error invoking conversational chain: {e}")
                        response = "Sorry, I encountered an error. Please try again or contact support@digidaratechnologies.com."
                    quick_replies = ['Enquire Now', 'quit']
            elif state == 'collect_enquiry':
                user_data['last_processed_time'] = save_user(user_data['name'], user_data['email'], user_data['phone'], user_data.get('course'), user_data.get('course_duration'), message=prompt, last_processed_time=user_data['last_processed_time'], llm=llm)
                response = "Thank you! Enquiry saved. How else can I assist? (Type 'quit' to exit)"
                user_data['step'] = 'free_chat'
                quick_replies = ['Enquire Now', 'quit']
    else:
        if state == 'welcome':
            response = "Hi! Welcome to DigiDara Technologies. I’m here to assist. Please type anything to start!"
        elif state in ['collect_name', 'collect_email', 'collect_phone', 'collect_course', 'collect_duration', 'free_chat', 'collect_enquiry']:
            response = f"Welcome back, {user_data.get('name', '')}! Continue or type anything. (Type 'quit' to exit)"

    user_chains[user_id] = user_data
    logger.info(f"Returning response: '{response}' with updated state: {user_data['step']}")
    return jsonify({'response': response, 'quickReplies': quick_replies})

if __name__ == "__main__":
    logger.warning("Running in development mode. Use Gunicorn for production.")
    app.run(debug=True, host='0.0.0.0', port=5000)
