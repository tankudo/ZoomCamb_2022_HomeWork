from locust import task
from locust import between
from locust import HttpUser

sample = [[6.4, 3.5, 4.5, 1.2]]

class MLZoomUser(HttpUser):

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)
