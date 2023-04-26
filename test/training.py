# imports
from fastapi.testclient import TestClient
from app.main import app

import base64

# creating Test Client
testClient = TestClient(app=app)


def test_post_request() -> None:
    data = {
        'filename': 'file.csv', 
        'content': base64.b64encode(s='testing post request')
    }
    response = testClient.post(url='/training', content=data)
    assert response.status_code == 200
    assert response.json() == {'response': 'testing post request'}