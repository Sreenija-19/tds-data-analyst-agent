import requests

url = "https://tds-data-analyst-agent-duih.onrender.com/answer"
file_path = r"C:\Users\ADMIN\Downloads\git\tds-data-analyst-agent\questions.txt"

# Read questions from file
with open(file_path, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

# Prepare JSON body
data = {"questions": questions}

# Send POST request
response = requests.post(url, json=data)

print("Response status:", response.status_code)
print("Response content:", response.text)
