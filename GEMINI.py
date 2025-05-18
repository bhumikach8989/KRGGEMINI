from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, render_template_string

import os
import uuid
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import networkx as nx
import matplotlib.pyplot as plt
import re


load_dotenv()

# ✅ Define folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
GENERATED_FOLDER = os.path.join(BASE_DIR, 'generated')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)

# ✅ Securely get the API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY environment variable!")

# ✅ Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])


def extract_knowledge_triples(text):
    prompt = f"""
Make the text short with simplified sentences such that triples can be extracted easily and Respond only with a valid JSON array of triples. Do not include any explanations or text before or after and Extract knowledge triples from this text:
Format example: [
  {{"subject": "...", "predicate": "...", "object": "..."}}
]
Text:
\"\"\"{text}\"\"\"
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    cleaned_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        triples = json.loads(cleaned_text)
        return triples
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON from Gemini response:")
        print(raw_text)
        return []


def build_and_save_graph(triples, image_path):
    G = nx.DiGraph()
    for triple in triples:
        subj = triple.get("subject")
        pred = triple.get("predicate")
        obj = triple.get("object")
        if subj and pred and obj:
            G.add_edge(subj, obj, label=pred)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Knowledge Representation Graph")
    plt.savefig(image_path)
    plt.close()


@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>Knowledge Graph Generator</title>
  <style>
    body { font-family: Arial; padding: 20px; }
    .container { max-width: 600px; margin: auto; }
    img { max-width: 100%; margin-top: 20px; border: 1px solid #ccc; padding: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Generate Knowledge Graph from PDF</h2>
    <form id="uploadForm" enctype="multipart/form-data">
      <input type="file" name="pdf" accept="application/pdf" required>
      <br><br>
      <button type="submit">Upload and Generate</button>
    </form>

    <div id="response" style="margin-top:20px;"></div>
    <img id="graphImage" src="" style="display:none;" alt="Knowledge Graph">
  </div>

<script>
const form = document.getElementById('uploadForm');
const responseDiv = document.getElementById('response');
const graphImage = document.getElementById('graphImage');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  responseDiv.innerText = "Processing PDF and generating graph...";
  graphImage.style.display = "none";

  const formData = new FormData(form);
  const res = await fetch('/upload', { method: 'POST', body: formData });

  if (res.ok) {
    const data = await res.json();
    responseDiv.innerText = "Graph generated successfully:";
    graphImage.src = data.image_url + "?t=" + new Date().getTime();  // cache busting
    graphImage.style.display = "block";
  } else {
    responseDiv.innerText = "❌ Error generating graph.";
  }
});
</script>
</body>
</html>
""")


@app.route('/upload', methods=['POST'])
def upload_pdf():
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_filename = f"{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    pdf_file.save(pdf_path)

    text = extract_text_from_pdf(pdf_path)
    triples = extract_knowledge_triples(text)

    if not triples:
        return jsonify({"error": "No triples extracted from PDF"}), 500

    image_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join(GENERATED_FOLDER, image_filename)
    build_and_save_graph(triples, image_path)

    return jsonify({"image_url": f"/generated/{image_filename}"})


@app.route('/generated/<path:filename>')
def serve_static(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
