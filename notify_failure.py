"""
Failure Alert Email
====================
Sends a plain-text email via Gmail SMTP. Used by
.github/workflows/daily-run.yml as an `if: failure()` step -- it only runs
when the daily agent job has already failed, so it never touches or slows
down a normal successful run.

Needs two GitHub Actions secrets (see the workflow file for how to create
them):
  GMAIL_ADDRESS       -- the Gmail account sending the alert
  GMAIL_APP_PASSWORD  -- a Gmail App Password for that account (NOT its
                         normal login password)

Recipient, subject, and body come from environment variables so nothing
is hardcoded here.
"""

import os
import smtplib
from email.mime.text import MIMEText


def main():
    gmail_address = os.environ["GMAIL_ADDRESS"]
    gmail_app_password = os.environ["GMAIL_APP_PASSWORD"]
    to_address = os.environ["ALERT_TO_ADDRESS"]
    subject = os.environ["ALERT_SUBJECT"]
    body = os.environ["ALERT_BODY"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to_address

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, [to_address], msg.as_string())

    print(f"Failure alert emailed to {to_address}")


if __name__ == "__main__":
    main()
