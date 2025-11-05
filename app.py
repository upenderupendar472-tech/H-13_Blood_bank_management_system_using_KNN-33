from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
import math
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'blood_bank_secret_key_2024'

# Database initialization
def init_db():
    conn = sqlite3.connect('blood_bank.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Donors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            age INTEGER NOT NULL,
            location TEXT NOT NULL,
            last_donation_date TEXT,
            health_status TEXT DEFAULT 'Good',
            availability TEXT DEFAULT 'Available',
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            age INTEGER NOT NULL,
            location TEXT NOT NULL,
            units_needed INTEGER NOT NULL,
            urgency TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            request_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Blood inventory table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blood_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            blood_group TEXT NOT NULL,
            units_available INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default blood groups
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    for bg in blood_groups:
        cursor.execute('INSERT OR IGNORE INTO blood_inventory (blood_group, units_available) VALUES (?, ?)', (bg, 0))
    
    # Create default admin user
    admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
    cursor.execute('INSERT OR IGNORE INTO users (username, email, password, user_type) VALUES (?, ?, ?, ?)',
                  ('admin', 'admin@bloodbank.com', admin_password, 'admin'))
    
    conn.commit()
    conn.close()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# KNN Donor Matching Algorithm
class DonorMatcher:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    
    def prepare_features(self, donors_df):
        """Prepare features for KNN algorithm"""
        # Encode categorical variables
        categorical_columns = ['blood_group', 'location', 'health_status', 'availability']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            donors_df[col] = self.label_encoders[col].fit_transform(donors_df[col].astype(str))
        
        # Calculate days since last donation
        donors_df['last_donation_days'] = donors_df['last_donation_date'].apply(
            lambda x: (datetime.now() - datetime.strptime(x, '%Y-%m-%d')).days if x and x != 'None' else 365
        )
        
        # Select features for KNN
        features = ['blood_group', 'age', 'last_donation_days', 'latitude', 'longitude']
        return donors_df[features]
    
    def find_matching_donors(self, patient_data, donors_df, k=5):
        """Find k nearest donors for a patient"""
        try:
            # Prepare donor features
            donor_features = self.prepare_features(donors_df.copy())
            
            # Fit KNN model
            self.knn.fit(donor_features)
            
            # Prepare patient features
            patient_features = self.prepare_features(pd.DataFrame([patient_data]))
            
            # Find nearest neighbors
            distances, indices = self.knn.kneighbors(patient_features)
            
            # Get matching donors
            matching_donors = donors_df.iloc[indices[0]]
            matching_donors['distance_score'] = distances[0]
            
            return matching_donors.to_dict('records')
        
        except Exception as e:
            print(f"Error in KNN matching: {e}")
            return []

# Helper functions
def get_db_connection():
    conn = sqlite3.connect('blood_bank.db')
    conn.row_factory = sqlite3.Row
    return conn

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def update_blood_inventory(blood_group, units_change):
    """Update blood inventory"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'UPDATE blood_inventory SET units_available = units_available + ?, last_updated = CURRENT_TIMESTAMP WHERE blood_group = ?',
        (units_change, blood_group)
    )
    
    conn.commit()
    conn.close()

# Authentication decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, hash_password(password))
        ).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_type'] = user['user_type']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO users (username, email, password, user_type) VALUES (?, ?, ?, ?)',
                (username, email, hash_password(password), user_type)
            )
            
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
    
    return render_template('register.html')

@app.route('/donor/register', methods=['GET', 'POST'])
@login_required
def donor_register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            blood_group = request.form['blood_group']
            age = int(request.form['age'])
            location = request.form['location']
            last_donation = request.form.get('last_donation') or None
            health_status = request.form.get('health_status', 'Good')
            
            # Simple coordinate generation
            latitude = np.random.uniform(12.0, 13.0)
            longitude = np.random.uniform(77.0, 78.0)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO donors (user_id, name, email, phone, blood_group, age, location, 
                                 last_donation_date, health_status, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], name, email, phone, blood_group, age, location, 
                  last_donation, health_status, latitude, longitude))
            
            conn.commit()
            conn.close()
            
            flash('Donor registered successfully!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            flash(f'Error registering donor: {str(e)}', 'error')
    
    return render_template('donor_register.html')

@app.route('/patient/request', methods=['GET', 'POST'])
@login_required
def patient_request():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            phone = request.form['phone']
            blood_group = request.form['blood_group']
            age = int(request.form['age'])
            location = request.form['location']
            units_needed = int(request.form['units_needed'])
            urgency = request.form['urgency']
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO patients (user_id, name, email, phone, blood_group, age, location, units_needed, urgency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], name, email, phone, blood_group, age, location, units_needed, urgency))
            
            patient_id = cursor.lastrowid
            conn.commit()
            
            # Find matching donors using KNN
            donors_df = pd.read_sql('SELECT * FROM donors WHERE availability = "Available"', conn)
            
            if not donors_df.empty:
                matcher = DonorMatcher()
                
                # Generate patient coordinates
                patient_lat = np.random.uniform(12.0, 13.0)
                patient_lon = np.random.uniform(77.0, 78.0)
                
                patient_features = {
                    'blood_group': blood_group,
                    'age': age,
                    'location': location,
                    'last_donation_date': datetime.now().strftime('%Y-%m-%d'),
                    'latitude': patient_lat,
                    'longitude': patient_lon
                }
                
                matching_donors = matcher.find_matching_donors(patient_features, donors_df)
                
                # Update patient status
                cursor.execute('UPDATE patients SET status = ? WHERE id = ?', 
                             ('Matched' if matching_donors else 'No Match', patient_id))
                conn.commit()
                
                conn.close()
                
                flash(f'Request submitted successfully! Found {len(matching_donors)} potential donors.', 'success')
                return render_template('search_donors.html', donors=matching_donors, patient_blood_group=blood_group)
            else:
                conn.close()
                flash('No donors available at the moment.', 'warning')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error submitting request: {str(e)}', 'error')
    
    return render_template('patient_request.html')

@app.route('/search/donors')
def search_donors():
    blood_group = request.args.get('blood_group', '')
    location = request.args.get('location', '')
    
    conn = get_db_connection()
    
    query = 'SELECT * FROM donors WHERE availability = "Available"'
    params = []
    
    if blood_group:
        query += ' AND blood_group = ?'
        params.append(blood_group)
    
    if location:
        query += ' AND location LIKE ?'
        params.append(f'%{location}%')
    
    donors = conn.execute(query, params).fetchall()
    conn.close()
    
    return render_template('search_donors.html', donors=donors, search_blood_group=blood_group)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if session.get('user_type') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    
    # Get statistics
    total_donors = conn.execute('SELECT COUNT(*) FROM donors').fetchone()[0]
    total_patients = conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    blood_inventory = conn.execute('SELECT * FROM blood_inventory').fetchall()
    recent_requests = conn.execute('SELECT * FROM patients ORDER BY request_date DESC LIMIT 10').fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html', 
                         total_donors=total_donors,
                         total_patients=total_patients,
                         blood_inventory=blood_inventory,
                         recent_requests=recent_requests)

@app.route('/api/donors')
def api_donors():
    conn = get_db_connection()
    donors = conn.execute('SELECT * FROM donors').fetchall()
    conn.close()
    
    donors_list = [dict(donor) for donor in donors]
    return jsonify(donors_list)

@app.route('/api/inventory')
def api_inventory():
    conn = get_db_connection()
    inventory = conn.execute('SELECT * FROM blood_inventory').fetchall()
    conn.close()
    
    inventory_list = [dict(item) for item in inventory]
    return jsonify(inventory_list)

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)