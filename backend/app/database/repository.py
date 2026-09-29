"""
Database schema and models for Saarthi Government Services
SQLite database with tables for services, complaints, users, and audit logs
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SaarthiDatabase:
    """
    SQLite database manager for Saarthi with security-focused schema
    """
    
    def __init__(self, db_path: str = "saarthi.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.init_database()
        logger.info(f"Database initialized at: {db_path}")
    
    def connect(self):
        """Create database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def init_database(self):
        """Create all required tables"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # 1. Services Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS services (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    processing_time TEXT,
                    cost TEXT,
                    eligibility TEXT,
                    authority TEXT NOT NULL,
                    required_documents TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    languages_supported TEXT,
                    data_sensitivity TEXT DEFAULT 'low',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 2. Complaints Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id TEXT UNIQUE NOT NULL,
                    citizen_name TEXT NOT NULL,
                    citizen_email TEXT,
                    citizen_phone TEXT,
                    issue_category TEXT NOT NULL,
                    issue_description TEXT NOT NULL,
                    status TEXT DEFAULT 'registered',
                    priority TEXT DEFAULT 'normal',
                    assigned_to TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    data_hash TEXT
                )
            ''')
            
            # 3. Service Applications Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    application_id TEXT UNIQUE NOT NULL,
                    citizen_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    status TEXT DEFAULT 'submitted',
                    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expected_completion_date TIMESTAMP,
                    completed_date TIMESTAMP,
                    submitted_documents TEXT,
                    notes TEXT,
                    FOREIGN KEY (service_id) REFERENCES services(id)
                )
            ''')
            
            # 4. Users Table (anonymized for privacy)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT UNIQUE NOT NULL,
                    preferred_language TEXT DEFAULT 'English',
                    last_active TIMESTAMP,
                    session_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 5. Audit Log Table (for AI assurance)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_hash TEXT,
                    action_type TEXT NOT NULL,
                    service_accessed TEXT,
                    query_hash TEXT,
                    result TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    data_sensitivity TEXT,
                    details TEXT
                )
            ''')
            
            # 6. Knowledge Base Table (for service search)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    keyword_type TEXT,
                    language TEXT DEFAULT 'English',
                    relevance_score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (service_id) REFERENCES services(id)
                )
            ''')
            
            # 7. Query Cache Table (for performance optimization)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    original_query TEXT,
                    source_language TEXT,
                    matched_services TEXT,
                    intent TEXT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 8. Feedback Table (for continuous improvement)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    user_hash TEXT,
                    query_hash TEXT,
                    service_id TEXT,
                    rating INTEGER,
                    feedback_text TEXT,
                    is_helpful BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_service_category ON services(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_complaint_status ON complaints(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_complaint_date ON complaints(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_keyword ON knowledge_base(keyword, language)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_application_status ON service_applications(status)')
            
            conn.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_service(self, service_data: Dict) -> bool:
        """Add a service to the database"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO services 
                (id, name, description, category, processing_time, cost, 
                 eligibility, authority, required_documents, steps, 
                 languages_supported, data_sensitivity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                service_data['id'],
                service_data['name'],
                service_data['description'],
                service_data['category'],
                service_data.get('processing_time'),
                service_data.get('cost'),
                service_data.get('eligibility'),
                service_data['authority'],
                json.dumps(service_data.get('required_documents', [])),
                json.dumps(service_data.get('steps', [])),
                json.dumps(service_data.get('languages_supported', ['English'])),
                service_data.get('data_sensitivity', 'low')
            ))
            
            conn.commit()
            logger.info(f"Service added: {service_data['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding service: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_service_by_id(self, service_id: str) -> Optional[Dict]:
        """Retrieve service by ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving service: {str(e)}")
            return None
        finally:
            conn.close()
    
    def search_services_by_keyword(self, keyword: str, language: str = 'English') -> List[Dict]:
        """Search services by keyword"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Search in knowledge base for relevant services
            cursor.execute('''
                SELECT DISTINCT s.* FROM services s
                INNER JOIN knowledge_base kb ON s.id = kb.service_id
                WHERE (kb.keyword LIKE ? OR s.name LIKE ? OR s.description LIKE ?)
                AND (kb.language = ? OR s.languages_supported LIKE ?)
                ORDER BY kb.relevance_score DESC
            ''', (
                f'%{keyword}%',
                f'%{keyword}%',
                f'%{keyword}%',
                language,
                f'%{language}%'
            ))
            
            rows = cursor.fetchall()
            services = [dict(row) for row in rows]
            
            logger.info(f"Found {len(services)} services for keyword: {keyword}")
            return services
            
        except Exception as e:
            logger.error(f"Error searching services: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_services_by_category(self, category: str) -> List[Dict]:
        """Get all services in a category"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM services WHERE category = ?', (category,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error retrieving services by category: {str(e)}")
            return []
        finally:
            conn.close()
    
    def add_complaint(self, complaint_data: Dict) -> Tuple[bool, str]:
        """Add a complaint to the database"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            complaint_id = f"COMP_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO complaints 
                (complaint_id, citizen_name, citizen_email, citizen_phone, 
                 issue_category, issue_description, status, priority, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                complaint_id,
                complaint_data.get('citizen_name'),
                complaint_data.get('citizen_email'),
                complaint_data.get('citizen_phone'),
                complaint_data.get('issue_category'),
                complaint_data.get('issue_description'),
                'registered',
                complaint_data.get('priority', 'normal'),
                complaint_data.get('data_hash')
            ))
            
            conn.commit()
            logger.info(f"Complaint registered: {complaint_id}")
            return True, complaint_id
            
        except Exception as e:
            logger.error(f"Error adding complaint: {str(e)}")
            conn.rollback()
            return False, ""
        finally:
            conn.close()
    
    def get_complaint_by_id(self, complaint_id: str) -> Optional[Dict]:
        """Retrieve complaint by ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM complaints WHERE complaint_id = ?', (complaint_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving complaint: {str(e)}")
            return None
        finally:
            conn.close()
    
    def update_complaint_status(self, complaint_id: str, new_status: str, 
                               notes: Optional[str] = None) -> bool:
        """Update complaint status"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE complaints 
                SET status = ?, updated_at = CURRENT_TIMESTAMP, resolution_notes = ?
                WHERE complaint_id = ?
            ''', (new_status, notes, complaint_id))
            
            conn.commit()
            logger.info(f"Complaint {complaint_id} status updated to: {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating complaint: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def add_knowledge_base_entry(self, service_id: str, keyword: str, 
                                 language: str = 'English', 
                                 relevance_score: float = 1.0) -> bool:
        """Add keyword mapping to knowledge base"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO knowledge_base 
                (service_id, keyword, language, relevance_score)
                VALUES (?, ?, ?, ?)
            ''', (service_id, keyword.lower(), language, relevance_score))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge base entry: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def cache_query_result(self, query_hash: str, query_data: Dict) -> bool:
        """Cache query results for performance"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO query_cache 
                (query_hash, original_query, source_language, matched_services, intent, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                query_hash,
                query_data.get('original_query'),
                query_data.get('source_language'),
                json.dumps(query_data.get('matched_services', [])),
                query_data.get('intent'),
                query_data.get('confidence', 0.0)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error caching query: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_cached_query(self, query_hash: str) -> Optional[Dict]:
        """Retrieve cached query results"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM query_cache WHERE query_hash = ?
            ''', (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # Update access count
                cursor.execute('''
                    UPDATE query_cache 
                    SET accessed_count = accessed_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                ''', (query_hash,))
                conn.commit()
                
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached query: {str(e)}")
            return None
        finally:
            conn.close()
    
    def add_audit_log(self, log_data: Dict) -> bool:
        """Add entry to audit log for compliance and security"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO audit_logs 
                (user_hash, action_type, service_accessed, query_hash, result, 
                 ip_address, user_agent, data_sensitivity, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_data.get('user_hash'),
                log_data.get('action_type'),
                log_data.get('service_accessed'),
                log_data.get('query_hash'),
                log_data.get('result'),
                log_data.get('ip_address'),
                log_data.get('user_agent'),
                log_data.get('data_sensitivity'),
                json.dumps(log_data.get('details', {}))
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding audit log: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def add_user_feedback(self, feedback_data: Dict) -> bool:
        """Record user feedback for model improvement"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            feedback_id = f"FB_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO user_feedback 
                (feedback_id, user_hash, query_hash, service_id, rating, feedback_text, is_helpful)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id,
                feedback_data.get('user_hash'),
                feedback_data.get('query_hash'),
                feedback_data.get('service_id'),
                feedback_data.get('rating'),
                feedback_data.get('feedback_text'),
                feedback_data.get('is_helpful')
            ))
            
            conn.commit()
            logger.info(f"Feedback recorded: {feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def populate_sample_data(self):
        """Populate database with sample government services"""
        sample_services = [
            {
                'id': 'birth_certificate',
                'name': 'Birth Certificate',
                'description': 'Official document issued to record the birth of a child',
                'category': 'vital_records',
                'processing_time': '7-15 days',
                'cost': 'Free',
                'eligibility': 'Any child born in India',
                'authority': 'Municipal Corporation / Local Registrar',
                'required_documents': ['Hospital discharge certificate', 'Parent ID proof', 'Residential proof'],
                'steps': ['Visit municipal office', 'Submit application form', 'Provide supporting documents', 'Receive certificate'],
                'languages_supported': ['English', 'Tamil', 'Telugu', 'Kannada', 'Hindi'],
                'data_sensitivity': 'high'
            },
            {
                'id': 'vehicle_registration',
                'name': 'Vehicle Registration',
                'description': 'Register motor vehicles with transport authority',
                'category': 'transport',
                'processing_time': '3-7 days',
                'cost': '₹500-2000',
                'eligibility': 'Vehicle owner with proof of purchase',
                'authority': 'Regional Transport Office (RTO)',
                'required_documents': ['Proof of ownership', 'Invoice/Bill of sale', 'Insurance certificate', 'Emission test'],
                'steps': ['Fill registration form', 'Submit to RTO office', 'Pay registration fee', 'Receive certificate'],
                'languages_supported': ['English', 'Tamil', 'Telugu', 'Kannada', 'Hindi'],
                'data_sensitivity': 'medium'
            },
            {
                'id': 'driving_license',
                'name': 'Driving License',
                'description': 'Authorization to operate motor vehicles on public roads',
                'category': 'transport',
                'processing_time': '15-30 days',
                'cost': '₹200-500',
                'eligibility': 'Age 18+ years',
                'authority': 'Regional Transport Office (RTO)',
                'required_documents': ['Address proof', 'Age proof', 'Passport photo', 'Medical certificate'],
                'steps': ['Apply at RTO office', 'Pass written test', 'Pass driving test', 'Receive license'],
                'languages_supported': ['English', 'Tamil', 'Telugu', 'Kannada', 'Hindi'],
                'data_sensitivity': 'medium'
            },
            {
                'id': 'ration_card',
                'name': 'Ration Card',
                'description': 'Access to subsidized food grains and essential commodities',
                'category': 'social_welfare',
                'processing_time': '30-45 days',
                'cost': '₹100-300',
                'eligibility': 'Indian citizen with annual income < ₹10 lakhs',
                'authority': 'Public Distribution System (PDS)',
                'required_documents': ['Proof of residence', 'Income certificate', 'Family list', 'Photo ID'],
                'steps': ['Visit PDS office', 'Complete application form', 'Submit documents', 'Verification', 'Receive card'],
                'languages_supported': ['English', 'Tamil', 'Telugu', 'Kannada', 'Hindi'],
                'data_sensitivity': 'high'
            },
            {
                'id': 'passport',
                'name': 'Passport',
                'description': 'Travel document for international travel',
                'category': 'travel',
                'processing_time': '7-30 days',
                'cost': '₹1500-3500',
                'eligibility': 'Indian citizen',
                'authority': 'Ministry of External Affairs',
                'required_documents': ['Birth certificate', 'Address proof', 'Photo identity', 'Passport photo', 'Police verification'],
                'steps': ['Visit passport office', 'Complete application', 'Pay application fee', 'Attend interview', 'Receive passport'],
                'languages_supported': ['English', 'Hindi'],
                'data_sensitivity': 'high'
            }
        ]
        
        for service in sample_services:
            self.add_service(service)
            # Add keywords to knowledge base
            keywords = [service['name'].lower()] + service['description'].lower().split()[:5]
            for keyword in keywords:
                self.add_knowledge_base_entry(service['id'], keyword, 'English')
        
        logger.info("Sample data populated successfully")


# Database initialization function
def init_saarthi_db(db_path: str = "saarthi.db") -> SaarthiDatabase:
    """Initialize and return database instance"""
    db = SaarthiDatabase(db_path)
    
    # Populate with sample data if empty
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM services")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        db.populate_sample_data()
    
    return db


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and populate database
    db = init_saarthi_db("saarthi.db")
    
    # Test sample queries
    print("\n" + "="*80)
    print("SAARTHI DATABASE TESTS")
    print("="*80)
    
    # Test 1: Get service by ID
    print("\nTest 1: Get service by ID")
    service = db.get_service_by_id("birth_certificate")
    if service:
        print(f"✓ Found service: {service['name']}")
    
    # Test 2: Search services by keyword
    print("\nTest 2: Search services by keyword")
    results = db.search_services_by_keyword("certificate", "English")
    print(f"✓ Found {len(results)} services for 'certificate'")
    
    # Test 3: Get services by category
    print("\nTest 3: Get services by category")
    transport_services = db.get_services_by_category("transport")
    print(f"✓ Found {len(transport_services)} transport services")
    
    # Test 4: Add complaint
    print("\nTest 4: Add complaint")
    success, complaint_id = db.add_complaint({
        'citizen_name': 'John Doe',
        'citizen_email': 'john@example.com',
        'citizen_phone': '9876543210',
        'issue_category': 'road_damage',
        'issue_description': 'Pothole on main street'
    })
    if success:
        print(f"✓ Complaint registered: {complaint_id}")
    
    print("\n" + "="*80)
