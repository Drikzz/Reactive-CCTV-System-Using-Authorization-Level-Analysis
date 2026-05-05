import sys
from pathlib import Path
from fpdf import FPDF
import unicodedata

# Files to include (workspace-relative)
FILES = [
    "scripts/streamlit_app.py",
    "scripts/camera_config_streamlit.py",
    "scripts/combined_yolo_facenet_behavior.py",
    "scripts/combined_yolo_facenet_only.py",
    "face_recognition/Facenet/facenet_main.py",
    "face_recognition/Facenet/facenet_capture.py",
    "face_recognition/Facenet/facenet_train.py",
    "utils/authorization_manager.py",
    "utils/room_activity_logger.py",
    "utils/confirmation_manager.py",
    "utils/rtsp_config_manager.py",
    "utils/session_metrics.py",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "code_export.pdf"

class PDF(FPDF):
    def header(self):
        # Override default header (we'll add file headings manually)
        pass

pdf = PDF(unit='pt', format='A4')
pdf.set_auto_page_break(auto=True, margin=36)
pdf.set_font('Courier', size=9)

included = []
missing = []

for rel in FILES:
    p = (REPO_ROOT / rel)
    if not p.exists():
        missing.append(rel)
        continue
    included.append(rel)
    text = p.read_text(encoding='utf-8')

    # Add a title page for the file
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 18, f"File: {rel}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font('Courier', size=9)

    # Write code lines with explicit cursor movement so each line advances correctly.
    for line in text.splitlines():
        # Replace tabs with 4 spaces
        line = line.replace('\t', '    ')
        # Normalize to ASCII-safe text (drop non-ASCII characters)
        line_safe = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore').decode('ascii')
        # Use width=0 (to right margin) and force next line after each call.
        pdf.multi_cell(0, 12, line_safe, new_x="LMARGIN", new_y="NEXT")

# Add summary page
pdf.add_page()
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 18, 'Export summary', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', size=10)
pdf.multi_cell(0, 14, f'Included files ({len(included)}):\n' + '\n'.join(included))
if missing:
    pdf.ln(6)
    pdf.set_text_color(255, 0, 0)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 14, f'Missing files ({len(missing)}):', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 12, '\n'.join(missing))

pdf.output(str(OUTPUT))
print(f"PDF generated: {OUTPUT}")
if missing:
    print("Missing files:")
    for m in missing:
        print(f" - {m}")
