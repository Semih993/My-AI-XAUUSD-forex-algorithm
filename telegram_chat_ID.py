import requests

TELEGRAM_TOKEN = "" # Enter telegram token here
url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
response = requests.get(url)
print(response.json())

