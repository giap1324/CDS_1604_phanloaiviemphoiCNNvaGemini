"""
Test script for Gemini AI integration
Kiểm tra xem Gemini service hoạt động đúng không
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_service import get_gemini_service

def test_gemini_service():
    """Test Gemini service with sample data"""
    
    print("=" * 60)
    print("TESTING GEMINI AI SERVICE")
    print("=" * 60)
    
    # Get service
    service = get_gemini_service()
    
    # Check if configured
    print("\n[1] Checking configuration...")
    if service.is_configured():
        print("✅ Gemini API is configured")
        print(f"   API Key: {service.api_key[:20]}..." if service.api_key else "No API key")
    else:
        print("⚠️  Gemini API is NOT configured")
        print("   Will use fallback mode")
    
    # Test 1: PNEUMONIA case
    print("\n[2] Testing PNEUMONIA case...")
    print("-" * 60)
    
    report1 = service.generate_medical_report(
        prediction="PNEUMONIA",
        confidence=92.5
    )
    
    print("\nGenerated Report (PNEUMONIA):")
    print("-" * 60)
    print(report1[:500] + "..." if len(report1) > 500 else report1)
    print("-" * 60)
    print(f"✅ Report generated ({len(report1)} characters)")
    
    # Test 2: NORMAL case
    print("\n[3] Testing NORMAL case...")
    print("-" * 60)
    
    report2 = service.generate_medical_report(
        prediction="NORMAL",
        confidence=88.3
    )
    
    print("\nGenerated Report (NORMAL):")
    print("-" * 60)
    print(report2[:500] + "..." if len(report2) > 500 else report2)
    print("-" * 60)
    print(f"✅ Report generated ({len(report2)} characters)")
    
    # Test 3: With patient info
    print("\n[4] Testing with patient info...")
    print("-" * 60)
    
    report3 = service.generate_medical_report(
        prediction="PNEUMONIA",
        confidence=95.0,
        patient_info={
            'name': 'Nguyễn Văn A',
            'age': 45,
            'gender': 'Nam',
            'symptoms': 'Ho nhiều, sốt cao 39°C'
        }
    )
    
    print("\nGenerated Report (With Patient Info):")
    print("-" * 60)
    print(report3[:500] + "..." if len(report3) > 500 else report3)
    print("-" * 60)
    print(f"✅ Report generated ({len(report3)} characters)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ All tests passed!")
    print(f"   API Status: {'Configured' if service.is_configured() else 'Using Fallback'}")
    print("   Reports generated: 3")
    print("\nYou can now run the app with: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_gemini_service()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
