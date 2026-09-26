"""
Quick test to verify AI backend is working
"""
import requests
import json

print("Testing AI Backend...")
print("=" * 50)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    response = requests.get("http://localhost:5000/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print("   ✅ Health check passed!")
except Exception as e:
    print(f"   ❌ Health check failed: {e}")

# Test 2: Translation
print("\n2. Testing /translate endpoint...")
try:
    response = requests.post(
        "http://localhost:5000/translate",
        json={
            "text": "Hello, how are you?",
            "source_lang": "en",
            "target_lang": "pa"
        }
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Input: Hello, how are you?")
    print(f"   Translation: {data.get('translation')}")
    print("   ✅ Translation test passed!")
except Exception as e:
    print(f"   ❌ Translation test failed: {e}")

print("\n" + "=" * 50)
print("If both tests passed, AI backend is working!")
print("The issue is likely browser cache.")
print("\nIn your browser, press: Ctrl + Shift + Delete")
print("Then clear cache and reload the page.")
