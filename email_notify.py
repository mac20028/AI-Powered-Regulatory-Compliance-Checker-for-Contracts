# email_notifier.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import os


def send_email_notification(
    smtp_server,
    smtp_port,
    username,
    password,
    to_email,
    subject,
    body,
    attachment_path=None
):
    """
    Send an email notification with a PDF attachment.
    """

    msg = MIMEMultipart()
    msg["From"] = username
    msg["To"] = to_email
    msg["Subject"] = subject

    # Add body text
    msg.attach(MIMEText(body, "plain"))

    # -------------------------
    # Attach PDF if provided
    # -------------------------
    if attachment_path:
        attachment_path = Path(attachment_path)

        if attachment_path.exists():
            with open(attachment_path, "rb") as file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={attachment_path.name}"
                )
                msg.attach(part)
        else:
            print(f"⚠ PDF not found at: {attachment_path}")

    try:
        server = smtplib.SMTP(smtp_server, int(smtp_port))
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent to {to_email}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def notify_compliance_update(
    pdf_path: Path,
    jurisdiction="Global",
    compliance_type="GDPR/HIPAA",
    recipient_email=None
):
    """
    Wrapper to send compliance notification with attached PDF.
    Also returns PDF bytes so Streamlit can allow download.
    """

    if recipient_email is None:
        print("[WARN] No recipient email provided. Skipping notification.")
        return None

    uploaded_name = pdf_path.name if pdf_path else "Unknown File"

    # -------------------------------------
    # ✨ Professional email body formatting
    # -------------------------------------
    body = (
        f"Hello,\n\n"
        f"The contract '{uploaded_name}' has been successfully updated for "
        f"{compliance_type} compliance under the jurisdiction '{jurisdiction}'.\n\n"
        f"The updated compliance PDF is attached to this email.\n"
        f"Please review the amended clauses at your convenience.\n\n"
        f"Regards,\n"
        f"Compliance Bot"
    )

    # ------------------------------
    # Load SMTP ENV variables
    # ------------------------------
    SMTP_SERVER = os.environ.get("SMTP_SERVER")
    SMTP_PORT = os.environ.get("SMTP_PORT")
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD]):
        print("❌ ERROR: Missing SMTP credentials in environment variables.")
        print("Expected: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD")
        return None

    # ----------------------------------
    # Send email with the updated PDF
    # ----------------------------------
    send_email_notification(
        smtp_server=SMTP_SERVER,
        smtp_port=SMTP_PORT,
        username=SMTP_USERNAME,
        password=SMTP_PASSWORD,
        to_email=recipient_email,
        subject=f"Contract Compliance Update: {uploaded_name}",
        body=body,
        attachment_path=pdf_path
    )

    # ----------------------------------------------------
    # Return PDF bytes to Streamlit for download button
    # ----------------------------------------------------
    if pdf_path.exists():
        return pdf_path.read_bytes()

    return None
