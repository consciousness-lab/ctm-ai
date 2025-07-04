# Deploying Demo to an EC2 Server

This guide walks you through setting up your EC2 server for both the frontend and backend of your demo application.

---

## Prerequisites

- **EC2 Instance:** A t2.micro instance is sufficient for this demo.
- **Security Group:** Ensure that your security group inbound rules allow the necessary ports:
  - **Port 80** for HTTP (Nginx)
  - **Port 443** for HTTPS (Nginx)
  - **Port 5000** for the backend (custom TCP with source `0.0.0.0/0`)

- **EC2 software**: need to install poetry, nginx, npm, tmux on the server.

---

## Server Side Setup

1. **Launch Your EC2 Instance:**
   Create a t2.micro instance using your preferred AMI (e.g., Ubuntu).

2. **Configure Security Group Inbound Rules:**
   In the EC2 Management Console, go to **Security Groups** → **Inbound Rules** and add a rule:
   - **Type:** Custom TCP
   - **Port Range:** 5000
   - **Source:** 0.0.0.0/0
   - make sure you can visit URL like `http://18.224.61.142:5000/api/upload`that are shown as `Method Not Allowed` instead of keeping waiting to load

   - HTTP and HTTPS should also be set accordingly.

---

## Frontend Deployment

### Step 1: Build the Frontend

Open a terminal and run the following commands:

```bash
# Navigate to your frontend directory
cd ~/ctm-ai/frontend
sudo su
export NODE_OPTIONS=--openssl-legacy-provider

# Remove old build artifacts and lock files
rm -rf node_modules package-lock.json build

# Clear the npm cache (optional)
npm cache clean --force

# Reinstall dependencies
npm install

# Build the project
npm run build

# Move the react built things into /var/www, if the enginx root is under /home, then no need
cp -r /home/ubuntu/ctm-ai/frontend/build/* /var/www/html/
```

### Step 2: Configure Nginx for the Frontend

1. **Edit the Nginx Configuration:**
   Open the configuration file for your site:
   ```bash
   sudo nano /etc/nginx/sites-available/frontend
   ```
2. **Paste the Following Configuration:**

   ```nginx
   server {
       listen 80;
       server_name 18.224.61.142;  # Replace with your domain if needed

       # Path to your built frontend files
       root /home/ubuntu/ctm-ai/frontend/build;
       index index.html;

       # Serve static files; fallback to index.html for SPA routing
       location / {
           try_files $uri $uri/ /index.html;
       }

       # Proxy API requests to the backend server on port 5000
       location /api {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Set Correct Permissions:**
   Ensure Nginx can read your frontend files. If necessary, change the ownership and permissions:
   ```bash
   sudo chown -R www-data:www-data path-to-dir/ctm-ai/frontend
   sudo chmod -R 755 path-to-dir/ctm-ai/frontend
   sudo chmod o+x /home/ec2-user
   sudo chmod o+x /home/ec2-user/tiny-scientist
   sudo chmod o+x /home/ec2-user/tiny-scientist/frontend
   sudo chmod o+x /home/ec2-user/tiny-scientist/frontend/build
   ```
   *(If you want to deploy from your current directory, adjust the root path accordingly.)*

4. **Test and Reload Nginx:**
   Test the configuration and reload Nginx:
   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```

---

## Backend Deployment

### Step 1: Install Dependencies

Navigate to your backend directory and install dependencies using Poetry:

```bash
cd ~/ctm-ai/backend
poetry install
```

### Step 2: Run the Backend

Start your backend server with Gunicorn:

```bash
export GOOGLE_API_KEY=xxx # for language and vision
export OPENAI_API_KEY=xxx # for search
export GOOGLE_CSE_ID=xxx # for search
export GEMINI_API_KEY=xxx # for audio
export DASHSCOPE_API_KEY=xxx # for code
poetry run gunicorn app:app --bind 0.0.0.0:5000
```

> **Note:**
> Ensure that your `app.py` exposes a callable WSGI application. For example, if you use a wrapper, set it as follows:
>
> ```python
> from app_wrapper import FlaskAppWrapper
> from flask_cors import CORS
>
> flask_wrapper = FlaskAppWrapper()
> CORS(flask_wrapper.app, origins=['http://localhost:3000', 'http://18.224.61.142'])
>
> # Expose the underlying Flask app as the WSGI application
> app = flask_wrapper.app
>
> if __name__ == '__main__':
>  app.run(port=5000, debug=True)
> ```

---

## Final Verification

- **Frontend:**
  Open your browser and navigate to `http://18.224.61.142`. Verify that the application loads correctly and static assets are served without redirection errors.

- **Backend:**
  Use tools like `curl`, Postman, or your browser’s network tab to test the API endpoints (e.g., `http://18.224.61.142/api/upload`).

- **Logs:**
  Check Nginx and backend logs for any errors:
  ```bash
  sudo tail -f /var/log/nginx/error.log
  ```

---

This guide should help you deploy your demo to an EC2 server, with a fresh build of your frontend served by Nginx and your backend running on Gunicorn. Let me know if you have any further questions or need additional adjustments!
