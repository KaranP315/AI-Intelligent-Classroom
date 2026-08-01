from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def add_slide(c, title, bullets):
    c.setFont("Helvetica-Bold", 28)
    c.drawString(1 * inch, 7.5 * inch, title)
    
    # Draw a line under title
    c.setLineWidth(2)
    c.setStrokeColorRGB(0.1, 0.4, 0.6) # Similar to the blue line in the template
    c.line(1 * inch, 7.3 * inch, 10 * inch, 7.3 * inch)
    
    c.setFont("Helvetica", 18)
    y = 6.8 * inch
    for bullet in bullets:
        # Simple text wrapping for bullets
        words = bullet.split()
        line = "• "
        for word in words:
            if c.stringWidth(line + word + " ", "Helvetica", 18) > 8.5 * inch:
                c.drawString(1.2 * inch, y, line)
                y -= 0.35 * inch
                line = "  " + word + " "
            else:
                line += word + " "
        c.drawString(1.2 * inch, y, line)
        y -= 0.5 * inch
    
    c.showPage()

def generate_pdf():
    c = canvas.Canvas("Project_Slides.pdf", pagesize=landscape(letter))
    
    # Slide 1
    bullets1 = [
        "Dual-Stream Pipeline: Built a system that simultaneously watches and listens to capture a complete picture.",
        "Visual Stream (Geometry over CNNs): Used MediaPipe Pose to track 5 specific 3D landmarks (nose, ears,",
        "shoulders) and applied simple math to determine if a student is looking up, looking down, or turning away.",
        "Audio Stream (The Double Gate): Audio is transcribed via Whisper and passes through two gates.",
        "1. Acoustic Gate: SSCon classifier (DialoGPT + XGBoost) checks if the tone is 'reasoning'.",
        "2. Semantic Gate: Sentence-Transformer checks if speech matches the lesson using Cosine Similarity.",
        "Time-Window Aggregation: Data is chunked into 1-minute blocks. The final score blends physical",
        "posture (60%) and verbal participation (40%) to ensure the system cannot be tricked."
    ]
    add_slide(c, "Methodology / Approach", bullets1)
    
    # Slide 2
    bullets2 = [
        "Core Architecture: Developed a multi-threaded Python runner handling the camera pipeline",
        "and microphone recording concurrently without bottlenecking.",
        "System Upgrades: Replaced a faulty CNN posture classifier with a robust 3D landmarking approach.",
        "Object Detection Overrides: Integrated YOLOv8n to scan for distractions like cell phones and bottles.",
        "Automated Teacher Feedback: Custom scoring script analyzes dialogue distribution to generate",
        "personalized feedback strings (e.g., 'Highly Inquisitive' or 'Passive Participant').",
        "Tech Stack: Python, OpenCV, Google MediaPipe Pose, YOLOv8-nano (Ultralytics), OpenAI Whisper,",
        "Sentence Transformers (all-MiniLM-L6-v2), Hugging Face (DialoGPT), and XGBoost."
    ]
    add_slide(c, "Work Done / Tools technologies used", bullets2)
    
    # Slide 3
    bullets3 = [
        "Solving the 'Distracted Debater': Our tests proved the semantic gate works. A student asking off-topic",
        "questions gets zero audio points, whereas a student explaining a concept gets full marks.",
        "MediaPipe > CNNs: Geometric tracking with MediaPipe was observed to be significantly more robust",
        "at catching subtle head turns than our previous image-based CNN approach.",
        "Output Profiles: The final terminal reports successfully produce distinct behavioral profiles.",
        "Demonstration Example: During live testing, two students had identical physical posture scores (80%).",
        "However, our multimodal blend correctly scored the on-topic student at 88/100 and the",
        "off-topic student at 48/100."
    ]
    add_slide(c, "Results / Observations", bullets3)
    
    # Slide 4
    bullets4 = [
        "Speaker Diarization: Replace our current 'last-visible student' shortcut with dedicated",
        "voice-recognition models to accurately track who is speaking, even if they are off-camera.",
        "Dynamic Semantic Thresholds: Automatically adjust the 0.35 similarity threshold based on the",
        "subject (e.g., math classes might require more exact phrasing than literature classes).",
        "Hardware Expansion: Scale the system to support multi-camera and multi-microphone arrays to",
        "cover large university lecture halls.",
        "Live Web Dashboard: Build a real-time web interface so teachers can get live behavioral alerts",
        "during the lecture, rather than just reading a post-class CSV report."
    ]
    add_slide(c, "Future Scope", bullets4)
    
    c.save()

if __name__ == "__main__":
    generate_pdf()
    print("PDF generated successfully.")
