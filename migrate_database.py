"""
Script to migrate database schema for Gemini AI integration
Thêm các trường medical_report và report_generated_at vào bảng diagnoses
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import Diagnosis

def migrate_database():
    """Migrate database to add new fields for medical reports"""
    
    print("=" * 60)
    print("DATABASE MIGRATION - Gemini AI Integration")
    print("=" * 60)
    
    with app.app_context():
        try:
            # Check if we need to migrate
            print("\n[1/4] Checking current database schema...")
            
            # Check if database file exists and has data
            db_path = 'instance/pneumonia_diagnosis.db'
            if os.path.exists(db_path):
                print(f"   Found existing database: {db_path}")
                
                # Try to check if new fields exist using raw SQL
                try:
                    from sqlalchemy import text
                    result = db.session.execute(text("PRAGMA table_info(diagnoses)"))
                    columns = [row[1] for row in result]
                    
                    if 'medical_report' in columns and 'report_generated_at' in columns:
                        print("✅ Database already has new fields. No migration needed.")
                        return True
                    else:
                        print("⚠️  New fields not found. Migration needed.")
                except Exception as e:
                    print(f"⚠️  Could not check schema: {e}")
                    print("   Will proceed with migration.")
            else:
                print("ℹ️  No existing database found. Will create new schema.")
            
            # Ask for confirmation
            print("\n[2/4] Migration will add these fields:")
            print("   - medical_report (Text)")
            print("   - report_generated_at (DateTime)")
            print("\n⚠️  WARNING: This will recreate all tables!")
            print("   All existing data will be LOST!")
            
            response = input("\nContinue? (yes/no): ").strip().lower()
            
            if response != 'yes':
                print("\n❌ Migration cancelled by user.")
                return False
            
            # Backup warning
            print("\n[3/4] Starting migration...")
            print("⚠️  Make sure you have backed up your data!")
            input("Press Enter to continue...")
            
            # Drop all tables
            print("\n   Dropping old tables...")
            db.drop_all()
            print("   ✅ Old tables dropped")
            
            # Create new tables with updated schema
            print("   Creating new tables...")
            db.create_all()
            print("   ✅ New tables created")
            
            # Verify
            print("\n[4/4] Verifying migration...")
            
            # Create a test record
            test_record = Diagnosis(
                filename='test_migration.jpg',
                filepath='/test/path.jpg',
                prediction='NORMAL',
                confidence=95.5,
                timestamp=datetime.now(),
                medical_report='Test report',
                report_generated_at=datetime.now()
            )
            
            db.session.add(test_record)
            db.session.commit()
            
            # Query back
            verified = Diagnosis.query.filter_by(filename='test_migration.jpg').first()
            
            if verified and verified.medical_report == 'Test report':
                print("   ✅ Migration verified successfully")
                
                # Clean up test record
                db.session.delete(verified)
                db.session.commit()
                print("   ✅ Test record cleaned up")
                
                print("\n" + "=" * 60)
                print("✅ DATABASE MIGRATION COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print("\nYou can now:")
                print("1. Set your GEMINI_API_KEY environment variable")
                print("2. Run the application: python app.py")
                print("3. Upload X-ray images to get AI-generated reports")
                print("\n" + "=" * 60)
                
                return True
            else:
                print("   ❌ Migration verification failed")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during migration: {e}")
            import traceback
            traceback.print_exc()
            return False

def rollback_database():
    """Rollback to old schema (without medical report fields)"""
    
    print("=" * 60)
    print("DATABASE ROLLBACK")
    print("=" * 60)
    print("\n⚠️  This will remove medical report functionality")
    print("   and delete all data!")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Rollback cancelled.")
        return False
    
    with app.app_context():
        try:
            db.drop_all()
            db.create_all()
            print("\n✅ Database rolled back successfully")
            return True
        except Exception as e:
            print(f"\n❌ Error during rollback: {e}")
            return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rollback':
        rollback_database()
    else:
        migrate_database()
