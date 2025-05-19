import sqlite3
import hashlib
import secrets
import smtplib
import re
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta


class AuthenticationSystem:
    def __init__(self, db_path="auth.db"):
        """Initialize the authentication system with the SQLite database."""
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Set up the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')
        
        # Create reset_tokens table for password resets
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reset_tokens (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create username_recovery table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS username_recovery (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create login_attempts table for security
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY,
            username TEXT,
            ip_address TEXT,
            attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password, salt=None):
        """Hash a password with a salt for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        # Use a strong hashing algorithm with multiple iterations for security
        key = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        ).hex()
        return key, salt
    
    def validate_password_strength(self, password):
        """Ensure the password meets minimum security requirements."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        return True, "Password meets requirements"
    
    def validate_email(self, email):
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email):
            return True
        return False
    
    def register_user(self, username, email, password):
        """Register a new user in the database."""
        # Validate inputs
        if not username or not email or not password:
            return False, "All fields are required"
        
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        valid_password, password_msg = self.validate_password_strength(password)
        if not valid_password:
            return False, password_msg
        
        # Hash the password for storage
        password_hash, salt = self._hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, salt)
            )
            conn.commit()
            return True, "User registered successfully"
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return False, "Username already exists"
            elif "email" in str(e):
                return False, "Email already registered"
            return False, "Error registering user"
        finally:
            conn.close()
    
    def login(self, username, password, ip_address="127.0.0.1"):
        """Authenticate a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user information
        cursor.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        success = False
        user_id = None
        
        if user:
            user_id, stored_hash, salt = user
            # Verify the password
            calculated_hash, _ = self._hash_password(password, salt)
            success = (calculated_hash == stored_hash)
            
            if success:
                # Update last login time
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", 
                    (user_id,)
                )
        
        # Record the login attempt
        cursor.execute(
            "INSERT INTO login_attempts (username, ip_address, success) VALUES (?, ?, ?)",
            (username, ip_address, 1 if success else 0)
        )
        
        conn.commit()
        conn.close()
        
        if success:
            return True, user_id, "Login successful"
        return False, None, "Invalid username or password"
    
    def initiate_password_reset(self, email):
        """Create a password reset token and send it to the user's email."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find user by email
        cursor.execute("SELECT id, username FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Email not found"
        
        user_id, username = user
        
        # Generate a secure token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)  # Token valid for 24 hours
        
        # Save the token in the database
        cursor.execute(
            "INSERT INTO reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        
        conn.commit()
        conn.close()
        
        # In a real application, you would send an email here
        # This is a placeholder for the email sending logic
        email_sent = self._send_reset_email(email, username, token)
        
        if email_sent:
            return True, "Password reset instructions sent to your email"
        return False, "Error sending reset email"
    
    def reset_password(self, token, new_password):
        """Reset a user's password using a valid token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find the token and check if it's valid
        current_time = datetime.now()
        cursor.execute(
            "SELECT user_id, expires_at, used FROM reset_tokens WHERE token = ?", 
            (token,)
        )
        token_info = cursor.fetchone()
        
        if not token_info:
            conn.close()
            return False, "Invalid or expired token"
        
        user_id, expires_at, used = token_info
        expires_at = datetime.fromisoformat(expires_at)
        
        if used or current_time > expires_at:
            conn.close()
            return False, "Token has expired or already been used"
        
        # Validate new password
        valid_password, password_msg = self.validate_password_strength(new_password)
        if not valid_password:
            conn.close()
            return False, password_msg
        
        # Update the password
        password_hash, salt = self._hash_password(new_password)
        cursor.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (password_hash, salt, user_id)
        )
        
        # Mark the token as used
        cursor.execute(
            "UPDATE reset_tokens SET used = 1 WHERE token = ?",
            (token,)
        )
        
        conn.commit()
        conn.close()
        
        return True, "Password has been reset successfully"
    
    def initiate_username_recovery(self, email):
        """Send username recovery information to the user's email."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find user by email
        cursor.execute("SELECT id, username FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Email not found"
        
        user_id, username = user
        
        # Generate a token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        # Save the token
        cursor.execute(
            "INSERT INTO username_recovery (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        
        conn.commit()
        conn.close()
        
        # This would send an email with the username
        email_sent = self._send_username_email(email, username)
        
        if email_sent:
            return True, "Username recovery instructions sent to your email"
        return False, "Error sending recovery email"
    
    def _send_reset_email(self, email, username, token):
        """Send a password reset email to the user (placeholder function)."""
        # In a real application, you would configure SMTP settings
        # This is a placeholder that returns True to simulate successful sending
        reset_link = f"https://yourapp.com/reset-password?token={token}"
        
        # For demonstration purposes, print the email content
        print(f"""
        To: {email}
        Subject: Password Reset Instructions
        
        Hello {username},
        
        You requested a password reset. Click the link below to reset your password:
        {reset_link}
        
        If you didn't request this, please ignore this email.
        
        Best regards,
        Your App Team
        """)
        
        return True  # Return True to simulate successful email sending
    
    def _send_username_email(self, email, username):
        """Send a username recovery email to the user (placeholder function)."""
        # Similar to the reset email function, this is a placeholder
        print(f"""
        To: {email}
        Subject: Your Username Recovery
        
        Hello,
        
        You requested your username for our app.
        
        Your username is: {username}
        
        Best regards,
        Your App Team
        """)
        
        return True  # Return True to simulate successful email sending


# Streamlit UI Components
class StreamlitAuthUI:
    def __init__(self, auth_system):
        """Initialize with an authentication system instance."""
        self.auth = auth_system
        
    def login_page(self):
        """Display the login form."""
        st.title("Login")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col3, col2 = st.columns([1, 5, 1])
        
        with col1:
            if st.button("Login"):
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, user_id, message = self.auth.login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success(message)
                        st.rerun()  # Refresh the page
                    else:
                        st.error(message)
        
        with col2:
            if st.button("Sign Up"):
                st.session_state.page = "signup"
                st.rerun()
        
        # with st.expander("Forgot your credentials?"):
        #     tab1, tab2 = st.tabs(["Forgot Password", "Forgot Username"])
            
        #     with tab1:
        #         self.forgot_password_form()
            
        #     with tab2:
        #         self.forgot_username_form()
    
    def signup_page(self):
        """Display the sign up form."""
        st.title("Sign Up")
        
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if not username or not email or not password or not confirm_password:
                st.error("All fields are required")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = self.auth.register_user(username, email, password)
                if success:
                    st.success(message)
                    st.info("You can now log in with your credentials")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()
    
    def forgot_password_form(self):
        """Display the forgot password form."""
        email = st.text_input("Enter your email address", key="forgot_password_email")
        
        if st.button("Reset Password", key="reset_password_btn"):
            if not email:
                st.error("Please enter your email address")
            else:
                success, message = self.auth.initiate_password_reset(email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    def forgot_username_form(self):
        """Display the forgot username form."""
        email = st.text_input("Enter your email address", key="forgot_username_email")
        
        if st.button("Recover Username", key="recover_username_btn"):
            if not email:
                st.error("Please enter your email address")
            else:
                success, message = self.auth.initiate_username_recovery(email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    def reset_password_page(self, token):
        """Display the reset password form."""
        st.title("Reset Your Password")
        
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("Set New Password"):
            if not new_password or not confirm_password:
                st.error("Please enter and confirm your new password")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = self.auth.reset_password(token, new_password)
                if success:
                    st.success(message)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(message)

    # def dashboard(self):
    #     """Display the user dashboard after successful login."""
    #     st.title(f"Welcome, {st.session_state.username}!")
        
    #     st.write("You are now logged in to the system.")
        
    #     if st.button("Logout"):
    #         st.session_state.logged_in = False
    #         st.session_state.user_id = None
    #         st.session_state.username = None
    #         st.session_state.page = "login"
    #         st.rerun()
