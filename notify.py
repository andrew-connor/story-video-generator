"""
Notification helpers for the DBSM pipeline.
Plays a chime, sends a Gmail notification, and sends an ntfy.sh push notification
when a pipeline step completes.
"""

import os
import base64
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path

DEFAULT_NOTIFY_EMAIL = "andrew.callaghan.connor@gmail.com"
GMAIL_CLIENT_SECRETS_FILE = "gmail_client_secret.json"
GMAIL_TOKEN_FILE = "gmail_token.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

NTFY_TOPIC = "dbsm-notifications"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


def play_chime():
    """Play a completion chime. Uses winsound on Windows, silent fallback elsewhere."""
    try:
        import winsound
        winsound.Beep(880, 200)
        winsound.Beep(1100, 200)
        winsound.Beep(1320, 400)
    except ImportError:
        pass
    except Exception:
        pass


def get_authenticated_service():
    """Authenticate with Google OAuth, reusing saved token if available."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    import google_auth_oauthlib.flow
    import googleapiclient.discovery

    creds = None
    if os.path.exists(GMAIL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                GMAIL_CLIENT_SECRETS_FILE, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(GMAIL_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    gmail = googleapiclient.discovery.build("gmail", "v1", credentials=creds)
    return gmail


def send_notification_email(to_address: str, subject: str, body: str):
    """Send a notification email via Gmail API using OAuth. Fails silently."""
    try:
        gmail = get_authenticated_service()
        message = MIMEText(body)
        message["to"] = to_address
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()
        print(f"  Notification email sent to {to_address}")
    except Exception as e:
        print(f"  ⚠️  Could not send notification email: {e}")


def send_ntfy(subject: str, body: str):
    """Send a push notification via ntfy.sh. Fails silently."""
    try:
        # ntfy message body is kept short — just the subject line
        # Full details are in the email
        data = subject.encode("utf-8")
        req = urllib.request.Request(
            NTFY_URL,
            data=data,
            headers={
                "Title": subject,
                "Priority": "default",
                "Tags": "bell",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"  Push notification sent to ntfy.sh/{NTFY_TOPIC}")
    except Exception as e:
        print(f"  ⚠️  Could not send push notification: {e}")


def notify(subject: str, body: str, to_address: str = DEFAULT_NOTIFY_EMAIL):
    """Play chime, send push notification, and send email."""
    play_chime()
    send_ntfy(subject, body)
    if to_address:
        send_notification_email(to_address, subject, body)


def prompt_notify_email() -> str:
    """Prompt for notification email address, defaulting to the standard address."""
    choice = input(
        f"  Notification email (default: '{DEFAULT_NOTIFY_EMAIL}', or type address, or 'none' to skip): "
    ).strip()
    if choice.lower() == "none":
        return ""
    return choice if choice else DEFAULT_NOTIFY_EMAIL
