# Eight Sleep MLOps Assignment

## How to Run the Application

### 1. Build the model
Generate the model weights file before containerizing:
```bash
python src/com/eightsleep/main/create_model.py
```
Before proceeding to the next step please ensure to have docker installed on the system.

### 2. Build and start the application using Docker Compose
From the project root (where the Dockerfile and docker-compose.yml are located):
```bash
docker-compose up --build
```

This command:
Builds the Docker image.

Starts two containers:
1. server → FastAPI service (port 8000)
2. generator → event generator that streams events to the server.

### 3. Verify the service
Once both containers are running, open a new terminal and run:
```bash
curl http://localhost:8000/users/user-1/median
```
You can also try another command:
```bash 
curl http://localhost:8000/stats 
```
You should see output like:
```bash
{"user_id":"user-1","median":0.05179734155535698}
```
or

```bash
{"requests":29992,"ingested_events":299920,"uptime_sec":196.07,"users_tracked":10000,"rps":152.97,"avg_events_per_request":10.0,"total_scores_stored":299920,"overall_median_score":0.05024889390915632}
```

