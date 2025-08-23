#!/usr/bin/env python3
"""
Test script to verify all required packages are installed
"""

def test_imports():
    print("Testing imports...")
    
    try:
        from langchain_groq import ChatGroq
        print("✅ langchain_groq imported successfully")
    except ImportError as e:
        print(f"❌ langchain_groq failed: {e}")
    
    try:
        from langchain_core.tools import tool
        print("✅ langchain_core.tools imported successfully")
    except ImportError as e:
        print(f"❌ langchain_core.tools failed: {e}")
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests failed: {e}")
    
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        print("✅ DuckDuckGoSearchRun imported successfully")
    except ImportError as e:
        print(f"❌ DuckDuckGoSearchRun failed: {e}")
    
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        print("✅ langchain.agents imported successfully")
    except ImportError as e:
        print(f"❌ langchain.agents failed: {e}")
    
    try:
        from langchain import hub
        print("✅ langchain hub imported successfully")
    except ImportError as e:
        print(f"❌ langchain hub failed: {e}")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv failed: {e}")
    
    print("\nTesting tool initialization...")
    
    try:
        search_tool = DuckDuckGoSearchRun()
        print("✅ DuckDuckGoSearchRun tool created successfully")
    except Exception as e:
        print(f"❌ DuckDuckGoSearchRun tool failed: {e}")
    
    print("\nAll import tests completed!")

if __name__ == "__main__":
    test_imports()
