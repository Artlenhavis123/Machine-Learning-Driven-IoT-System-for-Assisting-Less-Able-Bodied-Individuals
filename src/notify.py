import os

def send_email_notification(subject, body, recipient):
    import smtplib
    from email.mime.text import MIMEText

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
    print("📧 Email sent!")


def send_sms_notification(body, recipient):
    from twilio.rest import Client

    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH")
    twilio_number = os.getenv("TWILIO_PHONE")

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=body,
        from_=twilio_number,
        to=recipient
    )
    print("📱 SMS sent!")

