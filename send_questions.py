import requests

url = "https://tds-data-analyst-agent-duih.onrender.com/answer"
file_path = r"C:\Users\ADMIN\Downloads\git\tds-data-analyst-agent\questions.txt"

with open(file_path, "rb") as f:
    files = {"questions": (file_path.split("\\")[-1], f, "text/plain")}
    response = requests.post(url, files=files)

# Print the JSON response
print(response.status_code)
print(response.json())
