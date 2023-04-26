# imports
from fastapi.testclient import TestClient
from app.main import app

import base64

# creating Test Client
testClient = TestClient(app=app)


def test_post_request() -> None:
    data = {
        'filename': 'file.csv', 
        'content': base64.b64encode(s=b'testing post request').decode(encoding='utf-8')
    }
    response = testClient.post(url='/training', json=data)
    assert response.status_code == 200
    assert response.json() == {'response': 'testing post request'}