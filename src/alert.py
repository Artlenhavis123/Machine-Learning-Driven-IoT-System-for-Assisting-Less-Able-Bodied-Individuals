from src.notify import send_email_notification, send_sms_notification
import os

def trigger_alert():
    method = os.getenv("NOTIFY_METHOD", "email")  # email or sms
    message = "⚠️ Fall detected by the system. Please check on the user immediately."

    if method == "email":
        recipient = os.getenv("EMAIL_RECIPIENT")
        send_email_notification("Fall Alert 🚨", message, recipient)
    elif method == "sms":
        recipient = os.getenv("SMS_RECIPIENT")
        send_sms_notification(message, recipient)
    else:
        print("❌ Unknown notification method.")
