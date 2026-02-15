import requests

TELEGRAM_TOKEN = "7605615062:AAEPIqm0ZmhUJ7o1S-GgPiK1WSRikgwmhKM"
url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
response = requests.get(url)
print(response.json())
