import os
from dotenv import load_dotenv
from stt import listen

import google.generativeai as genai


load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(listen())
print(response.text)
