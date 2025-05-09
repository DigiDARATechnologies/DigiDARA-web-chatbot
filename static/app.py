from flask import Flask,render_template
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Setup for SMTP
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    logger.error("EMAIL_ADDRESS or EMAIL_PASSWORD not found in environment variables.")
    raise ValueError("EMAIL_ADDRESS or EMAIL_PASSWORD not found in environment variables.")
MAIL_SERVER = 'smtpout.secureserver.net'
MAIL_PORT = 587  # Use 587 for TLS



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


app=Flask(__name__)
@app.route('/')
def freemasterclass():
    return render_template('freeclass.html')

if __name__=='__main__':
    app.run()
