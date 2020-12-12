from fastapi import FastAPI

from typing import Optional

app = FastAPI()

@app.get('/')
def get_root():
    return {'Hello': 'World'}
