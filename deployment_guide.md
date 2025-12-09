# Chroma-Mood Deployment Guide

## Architecture
- **Frontend**: Static HTML/JS hosted on Vercel. Captures media client-side.
- **Backend**: Flask API hosted on AWS EC2. Processes media using TensorFlow/OpenCV.

---

## Part 1: Backend Deployment (AWS EC2)

### 1. Launch EC2 Instance
1.  Log in to AWS Console -> EC2 -> Launch Instance.
2.  **Name**: `Chroma-Mood-Backend`.
3.  **OS Image**: Ubuntu Server 22.04 LTS (recommended).
4.  **Instance Type**: `t3.medium` (4GB RAM) or `t2.large`. **Do not use t2.micro** (1GB RAM is likely too small for TensorFlow).
5.  **Key Pair**: Create a new key pair (`chroma-key.pem`) and download it.
6.  **Network Settings**:
    *   Allow SSH traffic from Anywhere (0.0.0.0/0).
    *   Allow HTTP/HTTPS traffic.
    *   **Important**: Edit Security Group to add a Custom TCP Rule: **Port 5000**, Source: Anywhere (0.0.0.0/0).

### 2. Connect to EC2
Open your terminal (where you downloaded the key):
```bash
chmod 400 chroma-key.pem
ssh -i "chroma-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
```

### 3. Install System Dependencies
Run these commands on the EC2 instance:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git ffmpeg libsm6 libxext6
```

### 4. Clone Repository
```bash
git clone https://github.com/CodeHashe/Chroma-Mood.git
cd Chroma-Mood
```

### 5. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
```

### 6. Run the Application
For testing (keeps running only while terminal is open):
```bash
python3 app.py
```
*Note: Ensure `app.py` listens on `0.0.0.0`. If it says `127.0.0.1`, edit `app.py` to `app.run(host='0.0.0.0', port=5000)`.*

For production (keeps running in background):
```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

---

## Part 2: Frontend Deployment (Vercel)

### 1. Prepare Frontend Code
1.  Open `frontend/index.html` locally.
2.  Find the line:
    ```javascript
    const API_BASE_URL = "http://localhost:5000";
    ```
3.  Change it to your EC2 Public IP:
    ```javascript
    const API_BASE_URL = "http://<YOUR-EC2-PUBLIC-IP>:5000";
    ```
    *(Note: If you use Vercel (HTTPS), you might get "Mixed Content" errors connecting to HTTP EC2. For a quick test, you can disable "Block Insecure Content" in your browser settings for the Vercel site, OR set up SSL on EC2 using Nginx + Certbot).*

### 2. Deploy
1.  Go to [Vercel](https://vercel.com).
2.  Import your GitHub Repository.
3.  **Root Directory**: Select `frontend` (if you want to deploy just the frontend folder) OR keep root and configure Vercel to serve `frontend/index.html`.
    *   *Easier method*: Drag and drop the `frontend` folder to Netlify Drop.
4.  Deploy!

---

## Troubleshooting
- **Connection Refused**: Check EC2 Security Group (Port 5000 open?). Check if `gunicorn` is running (`ps aux | grep gunicorn`).
- **Mixed Content Error**: Browser blocks HTTP requests from HTTPS site.
    - *Fix*: Use `http://` for both (host frontend on S3 bucket without SSL).
    - *Fix*: Setup SSL on EC2 (requires a domain name).
