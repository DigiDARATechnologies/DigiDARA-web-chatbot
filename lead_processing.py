import pandas as pd
import gspread
import os
import logging
import json
from oauth2client.service_account import ServiceAccountCredentials
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import smtplib
from sqlalchemy import create_engine
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL Configuration using SQLAlchemy
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'Hemarahith1!'
MYSQL_HOST = 'localhost'
MYSQL_DATABASE = 'DigiDARA_Chat_db'
MYSQL_URI = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}'

# Create SQLAlchemy engine
engine = create_engine(MYSQL_URI)

# Google Sheets Setup
SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', SCOPE)
SHEET_NAME = 'LeadsData'

# Email Setup for SMTP
EMAIL_ADDRESS = 'support@digidaratechnologies.com'
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Add this to your .env file
MAIL_SERVER = 'smtpout.secureserver.net'
MAIL_PORT = 587  # Use 587 for TLS

# WhatsApp Setup
AISENSY_API_KEY = os.getenv("AISENSY_API_KEY")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# File to store last processed time
LAST_PROCESSED_FILE = 'last_processed.json'

def load_last_processed_time():
    """Load the last processed time from a file."""
    try:
        with open(LAST_PROCESSED_FILE, 'r') as f:
            data = json.load(f)
            return data.get('last_processed_time')
    except FileNotFoundError:
        logger.info("No last processed time file found. Starting fresh.")
        return None
    except Exception as e:
        logger.error("Error loading last processed time: %s", e)
        return None

def save_last_processed_time(last_processed_time):
    """Save the last processed time to a file."""
    try:
        with open(LAST_PROCESSED_FILE, 'w') as f:
            json.dump({'last_processed_time': last_processed_time}, f)
        logger.info("Saved last processed time: %s", last_processed_time)
    except Exception as e:
        logger.error("Error saving last processed time: %s", e)

try:
    client = gspread.authorize(CREDS)
    sheet = client.open(SHEET_NAME).sheet1
    logger.info("Google Sheets connection successful: %s", sheet.title)
except Exception as e:
    logger.error("Failed to connect to Google Sheets: %s", e)
    raise

# Fetch data from MySQL
def fetch_new_mysql_data(last_processed_time=None):
    query = "SELECT name, email, phone, message, course, date FROM users"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        logger.info("Fetched MySQL data: %s rows", len(df))
        logger.info("Raw MySQL data: %s", df.to_dict())
        if last_processed_time:
            if isinstance(last_processed_time, str):
                last_processed_time = datetime.strptime(last_processed_time, '%Y-%m-%d %H:%M:%S')
            df = df[df['date'] >= last_processed_time]
            logger.info("Filtered data greater than or equal to %s: %s rows", last_processed_time, len(df))
    return df, datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Classify lead using LLM
def classify_lead(message, llm):
    prompt = f"Classify the following message as a lead type (Hot, Warm, or Cold) based on interest in a course. Hot means strong interest, Warm means moderate interest, Cold means low or no interest. Message: '{message}' Lead type: "
    try:
        result = llm(prompt)
        lead_types = ['Hot', 'Warm', 'Cold']
        for lead_type in lead_types:
            if lead_type.lower() in result.lower():
                logger.info("Classified message '%s' as %s via LLM", message, lead_type)
                return lead_type
    except Exception as e:
        logger.error("LLM classification error: %s", e)

    message = message.lower()
    course_keywords = ['modules', 'course', 'details', 'covered', 'learn', 'training']
    disinterest_keywords = ['no', 'not', 'later', 'quit']
    interest_detected = any(keyword in message for keyword in course_keywords)
    disinterest_detected = any(keyword in message for keyword in disinterest_keywords)

    if interest_detected and not (disinterest_detected and 'quit' in message):
        logger.info("Classified message '%s' as Warm via keywords (course interest)", message)
        return 'Warm'
    elif disinterest_detected and 'quit' in message:
        logger.info("Classified message '%s' as Cold via keywords (disinterest with quit)", message)
        return 'Cold'
    elif any(keyword in message for keyword in ['urgent', 'need now', 'enroll', 'sign up']):
        logger.info("Classified message '%s' as Hot via keywords", message)
        return 'Hot'
    elif any(keyword in message for keyword in ['maybe', 'possibly', 'curious', 'interested in']):
        logger.info("Classified message '%s' as Warm via keywords", message)
        return 'Warm'
    logger.info("Classified message '%s' as Cold (default)", message)
    return 'Cold'

# Generate email template using LLM
def generate_email_template(name, lead_type, course, llm):
    prompt = f"""
Generate a professional, engaging email template for a lead interested in the {course} course at DigiDara Technologies. Include:
- A personalized greeting with the recipient's name.
- A brief introduction about DigiDara Technologies focusing on AI training.
- Highlight the {course} course with benefits: expert-led training, hands-on projects, flexible schedules, and placement support.
- Include detailed course content tailored to {course} (e.g., modules or topics covered).
- List other courses briefly: Python, Data Analysis, Data Science, GEN AI, Agentic AI, AI Digital Marketing, Agentic Setups.
- Add a strong call-to-action with a styled button (e.g., bold headers, colored buttons).
- Use HTML styling for headers, lists, and buttons (e.g., font-family: Arial, sans-serif).
- Keep it under 300 words.
"""
    try:
        # Simulate course content based on course name (to be enhanced with RAG if available)
        course_content = {
            "AI in Cloud Computing": "Covers cloud platforms (AWS, Azure), AI deployment, scalability, and real-time analytics.",
            "Python": "Includes basics, data structures, file handling, and object-oriented programming with projects.",
            "Data Analysis": "Focuses on pandas, NumPy, data visualization, and statistical analysis techniques.",
            "Data Science": "Encompasses machine learning, data modeling, and predictive analytics with hands-on datasets.",
            "GEN AI": "Explores generative models, NLP, and creative AI applications with practical exercises.",
            "Agentic AI": "Teaches autonomous agents, decision-making AI, and multi-agent systems with simulations.",
            "AI Digital Marketing": "Covers AI-driven campaigns, customer segmentation, and analytics tools.",
            "Agentic Setups": "Focuses on configuring agentic workflows, optimization, and deployment strategies."
        }.get(course, "Explore a variety of cutting-edge AI topics with hands-on practice.")

        if lead_type == 'Hot':
            return f"""
            <h2 style="color: #2E7D32; font-family: Arial, sans-serif;">Welcome, {name}!</h2>
            <p style="font-family: Arial, sans-serif;">We’re thrilled about your interest in our {course} course at DigiDara Technologies, where we excel in AI training!</p>
            <p style="font-family: Arial, sans-serif;">Benefits include:</p>
            <ul style="color: #424242; font-family: Arial, sans-serif;">
                <li>Expert-led training</li>
                <li>Hands-on projects</li>
                <li>Flexible schedules</li>
                <li>Placement support</li>
            </ul>
            <p style="font-family: Arial, sans-serif;">Course content: {course_content}</p>
            <p style="font-family: Arial, sans-serif;">Explore more: Python, Data Analysis, Data Science, GEN AI, Agentic AI, AI Digital Marketing, Agentic Setups.</p>
            <a href="https://digidaratechnologies.com/contact-us" style="background-color: #D32F2F; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-family: Arial, sans-serif; display: inline-block;">Enroll Now!</a>
            <p style="font-family: Arial, sans-serif;">Act fast—secure your spot today!</p>
            <p style="font-family: Arial, sans-serif;">Best,<br>DigiDARA Team</p>
            """
        elif lead_type == 'Warm':
            return f"""
            <h2 style="color: #0288D1; font-family: Arial, sans-serif;">Hi {name}, Discover {course}!</h2>
            <p style="font-family: Arial, sans-serif;">Thanks for your interest in {course} at DigiDara Technologies, your hub for AI training!</p>
            <p style="font-family: Arial, sans-serif;">Benefits include:</p>
            <ul style="color: #424242; font-family: Arial, sans-serif;">
                <li>Expert-led training</li>
                <li>Hands-on projects</li>
                <li>Flexible schedules</li>
                <li>Placement support</li>
            </ul>
            <p style="font-family: Arial, sans-serif;">Course content: {course_content}</p>
            <p style="font-family: Arial, sans-serif;">Other courses: Python, Data Analysis, Data Science, GEN AI, Agentic AI, AI Digital Marketing, Agentic Setups.</p>
            <a href="https://digidaratechnologies.com/contact-us" style="background-color: #0288D1; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-family: Arial, sans-serif; display: inline-block;">Book Your Demo</a>
            <p style="font-family: Arial, sans-serif;">Start your journey now!</p>
            <p style="font-family: Arial, sans-serif;">Best,<br>DigiDARA Team</p>
            """
        else:  # Cold
            return f"""
            <h2 style="color: #6D4C41; font-family: Arial, sans-serif;">Hello {name}, Explore {course}!</h2>
            <p style="font-family: Arial, sans-serif;">We’ve noted your interest in {course} at DigiDara Technologies, specializing in AI training.</p>
            <p style="font-family: Arial, sans-serif;">Benefits include:</p>
            <ul style="color: #424242; font-family: Arial, sans-serif;">
                <li>Expert trainers</li>
                <li>Hands-on learning</li>
                <li>Flexible schedules</li>
                <li>Placement support</li>
            </ul>
            <p style="font-family: Arial, sans-serif;">Course content: {course_content}</p>
            <p style="font-family: Arial, sans-serif;">Check out: Python, Data Analysis, Data Science, GEN AI, Agentic AI, AI Digital Marketing, Agentic Setups.</p>
            <a href="https://digidaratechnologies.com/contact-us" style="background-color: #6D4C41; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-family: Arial, sans-serif; display: inline-block;">Subscribe Now</a>
            <p style="font-family: Arial, sans-serif;">Stay tuned for updates!</p>
            <p style="font-family: Arial, sans-serif;">Best,<br>DigiDARA Team</p>
            """
    except Exception as e:
        logger.error("Error generating template: %s", e)
        return f"""
        <h2 style="color: #2E7D32; font-family: Arial, sans-serif;">Welcome, {name}!</h2>
        <p style="font-family: Arial, sans-serif;">Thanks for your interest in {course} at DigiDara Technologies! Contact us at digidaratechnologies@gmail.com or 9500406945.</p>
        <p style="font-family: Arial, sans-serif;">Best,<br>DigiDARA Team</p>
        """

# Write to Google Sheets
def write_to_google_sheets(df):
    try:
        client = gspread.authorize(CREDS)
        sheet = client.open(SHEET_NAME).sheet1
        headers = ['Name', 'Email', 'Phone', 'Message', 'Lead Type', 'Course', 'Date']
        logger.info("Attempting to write to Google Sheets with %s rows", len(df))
        if not df.empty:
            current_headers = sheet.row_values(1)
            if not current_headers or set(current_headers) != set(headers):
                sheet.clear()
                sheet.append_row(headers)
                logger.info("Updated headers in Google Sheets")
            for _, row in df.iterrows():
                row_data = [
                    row['name'],
                    row['email'],
                    row['phone'],
                    row['message'],
                    row['lead_type'],
                    row['course'],
                    row['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['date'], 'strftime') else str(row['date'])
                ]
                sheet.append_row(row_data)
                logger.info("Updated Google Sheets with lead data for %s", row['name'])
        else:
            logger.warning("No data to write to Google Sheets")
    except Exception as e:
        logger.error("Error writing to Google Sheets: %s", e)

# Send email using SMTP, optionally embedding an image at the bottom
def send_email(to_email, subject, body):
    # Create a multipart email
    message = MIMEMultipart()
    message['From'] = EMAIL_ADDRESS
    message['To'] = to_email
    message['Subject'] = subject

    # Path to the image file
    image_path = 'static/image/course_poster.jpg'
    image_attached = False

    # Attempt to load the image
    try:
        with open(image_path, 'rb') as img:
            image_data = img.read()
        image = MIMEImage(image_data, name=os.path.basename(image_path))
        image.add_header('Content-ID', '<course_poster>')
        message.attach(image)
        image_attached = True
        logger.info("Image %s loaded successfully for email to %s", image_path, to_email)
    except FileNotFoundError:
        logger.warning("Image file not found at %s. Sending email without image.", image_path)
    except Exception as e:
        logger.error("Error loading image for email to %s: %s", to_email, e)

    # Attach the HTML body, including the image reference only if the image was loaded
    if image_attached:
        body_with_image = f"""
        {body}
        <p style="text-align: center; margin-top: 20px;">
            <img src="cid:course_poster" alt="DigiDARA Course Poster" style="max-width: 100%; height: auto;">
        </p>
        """
        html_part = MIMEText(body_with_image, 'html')
    else:
        html_part = MIMEText(body, 'html')
    
    message.attach(html_part)

    # Send the email using SMTP
    try:
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()  # Enable TLS
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(message)
        logger.info("Email sent to %s", to_email)
    except smtplib.SMTPAuthenticationError as e:
        logger.error("SMTP authentication failed for %s: %s", to_email, e)
        raise
    except smtplib.SMTPException as e:
        logger.error("SMTP error while sending email to %s: %s", to_email, e)
        raise
    except Exception as e:
        logger.error("General error sending email to %s: %s", to_email, e)
        raise

# Send WhatsApp message via AiSensy API
def send_whatsapp_message(api_key, phone_number_id, to_phone, campaign_name, template_params):
    # Validate inputs
    if not all([api_key, phone_number_id, to_phone, campaign_name]):
        logger.error("Missing required WhatsApp parameters: api_key=%s, phone_number_id=%s, to_phone=%s, campaign_name=%s",
                     api_key, phone_number_id, to_phone, campaign_name)
        return False

    # Ensure all template parameters are strings and handle None/empty values
    template_params = [str(param).strip() if param is not None else "" for param in template_params]
    
    url = "https://backend.aisensy.com/campaign/t1/api/v2"
    headers = {"Content-Type": "application/json"}
    payload = {
        "apiKey": api_key,
        "campaignName": campaign_name,
        "destination": to_phone,
        "userName": "DigiDara Technologies",
        "source": phone_number_id,
        "templateParams": template_params,
        "tags": ["lead_processing"],
        "attributes": {"lead_source": "Chatbot"}
    }
    logger.debug(f"Sending WhatsApp payload: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            logger.info(f"WhatsApp message sent to {to_phone}")
            logger.info(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Failed to send WhatsApp message: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending WhatsApp message to {to_phone}: %s", e)
        return False

# Main workflow with MCP
def process_leads(mcp_context=None):
    logger.info("Starting process_leads function")
    if mcp_context and 'last_processed_time' in mcp_context:
        last_processed_time = mcp_context['last_processed_time']
    else:
        last_processed_time = None
    
    llm = mcp_context.get('llm') if mcp_context else None
    if not llm:
        raise ValueError("LLM instance not provided in MCP context")

    df, new_time = fetch_new_mysql_data(last_processed_time)
    logger.info("Processing leads with DataFrame: %s", df.to_dict() if not df.empty else "Empty")
    if not df.empty:
        # Classify: Determine lead type
        df['lead_type'] = df['message'].apply(lambda msg: classify_lead(msg, llm))
        logger.info("Classified leads: %s", df['lead_type'].tolist())

        # Save: Update Google Sheets
        write_to_google_sheets(df)

        # Generate and Send: Email and WhatsApp to the user
        for index, row in df.iterrows():
            try:
                email_template = generate_email_template(row['name'], row['lead_type'], row['course'], llm)
                send_email(row['email'], f"{row['course']} Opportunity", email_template)
                logger.info("Successfully sent email to lead: %s", row['name'])
            except Exception as e:
                logger.error("Failed to send email to lead %s: %s", row['name'], e)

            phone = row['phone']
            if not phone.startswith('+'):
                phone = f"+91{phone}"  # Assuming Indian numbers; adjust as needed
            logger.info("Prepared phone number for WhatsApp: %s", phone)

            # Send WhatsApp to the lead based on lead type
            try:
                if row['lead_type'] == 'Hot':
                    success = send_whatsapp_message(
                        AISENSY_API_KEY, 
                        WHATSAPP_PHONE_NUMBER_ID, 
                        phone, 
                        "DigiDARA Technologies Customer Classification - Hot", 
                        [row['name'], row['course']]
                    )
                elif row['lead_type'] == 'Warm':
                    success = send_whatsapp_message(
                        AISENSY_API_KEY, 
                        WHATSAPP_PHONE_NUMBER_ID, 
                        phone, 
                        "DigiDARA Technologies Customer Classification - Warm", 
                        [row['name'], row['course']]
                    )
                elif row['lead_type'] == 'Cold':
                    success = send_whatsapp_message(
                        AISENSY_API_KEY, 
                        WHATSAPP_PHONE_NUMBER_ID, 
                        phone, 
                        "DigiDARA Technologies Customer Classification - Cold", 
                        [row['name'], row['course']]
                    )
                if success:
                    logger.info("Successfully sent WhatsApp message to lead: %s", row['name'])
                else:
                    logger.error("Failed to send WhatsApp message to lead: %s", row['name'])
            except Exception as e:
                logger.error("Exception while sending WhatsApp message to lead %s: %s", row['name'], e)
    else:
        logger.warning("No new data to process")
    
    # Save the updated last processed time
    save_last_processed_time(new_time)
    return {'last_processed_time': new_time}

if __name__ == "__email__":
    from llm_service import get_llm
    logger.info("Running lead_processing as main module")
    process_leads({'llm': get_llm()})