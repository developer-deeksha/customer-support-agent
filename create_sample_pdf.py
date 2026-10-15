
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create the PDF inside the pdfs folder.
pdf_folder = Path("pdfs")
pdf_folder.mkdir(exist_ok=True)

pdf_path = pdf_folder / "cloudsync_support.pdf"

# This is the knowledge-base data that the agent will read.
knowledge_base = [
    (
        "CloudSync Pro Product Information",
        "CloudSync Pro supports real-time synchronization across "
        "5 devices, provides 1TB of storage, supports offline mode, "
        "and maintains version history for 30 days."
    ),
    (
        "Subscription Pricing",
        "The Basic plan costs $9 per month and includes 100GB storage "
        "and 2 devices. The Pro plan costs $19 per month and includes "
        "1TB storage and 5 devices. The Business plan costs $49 per "
        "month and includes 5TB storage and unlimited devices."
    ),
    (
        "Refund Policy",
        "Customers can request a refund within 14 days of the charge."
    ),
    (
        "Password Reset",
        "To reset a password, open the login page, select Forgot Password, "
        "and enter your email address. The reset link expires in 1 hour."
    ),
    (
        "Troubleshooting Sync Issues",
        "Check your internet connection and make sure the app is updated. "
        "Then sign out and sign in again. If the problem continues, "
        "contact customer support."
    ),
    (
        "Supported Platforms",
        "CloudSync Pro supports Windows 10 or later, macOS 12 or later, "
        "iOS 15 or later, Android 10 or later, and Linux Beta."
    ),
    (
        "Data Security",
        "CloudSync Pro uses AES-256 encryption at rest and in transit. "
        "The product is SOC 2 Type II certified and uses a "
        "zero-knowledge architecture."
    ),
    (
        "File Size Limits",
        "Individual files can be up to 10GB on Pro and Business plans "
        "and 2GB on the Basic plan."
    ),
]

styles = getSampleStyleSheet()
story = []

story.append(Paragraph("CloudSync Pro Customer Support Knowledge Base", styles["Title"]))
story.append(Spacer(1, 20))

for heading, content in knowledge_base:
    story.append(Paragraph(heading, styles["Heading2"]))
    story.append(Paragraph(content, styles["BodyText"]))
    story.append(Spacer(1, 14))

document = SimpleDocTemplate(
    str(pdf_path),
    pagesize=A4,
    title="CloudSync Pro Support Knowledge Base"
)

document.build(story)

print(f"PDF created successfully: {pdf_path.resolve()}")