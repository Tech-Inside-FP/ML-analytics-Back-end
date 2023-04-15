# imports
from fastapi.testclient import TestClient
from app.main import app


# creating Test Client
testClient = TestClient(app=app)

    
def test_root_response() -> None:
    response = testClient.get(url='/')
    assert response.status_code == 200
    assert response.json() == {'ML-AnAIytics': 'Running'}
