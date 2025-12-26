import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

def send_meeting_email(meeting_data: dict):
    subject = "📅 New Meeting Booked - Aryu Bot"

    body = f"""
New meeting has been booked.

Name: {meeting_data['name']}
Phone: {meeting_data['phone']}
Email: {meeting_data['email']}
Preferred Time: {meeting_data['datetime']}
Purpose: {meeting_data.get('purpose', '-')}

Session ID: {meeting_data.get('session_id')}
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USERNAME
    msg["To"] = ADMIN_EMAIL

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
