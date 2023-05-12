import os

def fastmode():
    fast = os.getenv("FAST")
    return fast and fast.lower() == "true"
