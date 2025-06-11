import requests
import json

url = "http://127.0.0.1:5000/api"

headers = {
    "Content-Type": "application/json",
    "Authorization": "SuaChave"
}

data = {
    "consulta": "Onde localiza a granja?"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Status code:", response.status_code)
print("Resposta:", response.text)
