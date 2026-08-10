import sys
import os
import json
import websocket

# Find the src directory
CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from authenticator import get_websocket_url

SYMBOL = "R_100"


def main():

    print("Getting WebSocket URL...")

    url = get_websocket_url()

    print("URL obtained.")
    print()

    ws = websocket.WebSocket()

    print("Connecting...")

    ws.connect(url)

    print("Connected!")
    print()

    request = {
        "ticks": SYMBOL,
        "subscribe": 1
    }

    print("Request:")
    print(request)
    print()

    ws.send(json.dumps(request))

    response = json.loads(ws.recv())

    print("Response:")
    print(response)

    ws.close()


if __name__ == "__main__":
    main()

SYMBOL = "R_100"


def main():

    print("Getting WebSocket URL...")

    url = get_websocket_url()

    print("URL obtained.")
    print()

    ws = websocket.WebSocket()

    print("Connecting...")

    ws.connect(url)

    print("Connected!")
    print()

    request = {
        "ticks": SYMBOL,
        "subscribe": 1
    }

    print("Request:")
    print(request)
    print()

    ws.send(json.dumps(request))

    response = json.loads(ws.recv())

    print("Response:")
    print(response)

    ws.close()


if __name__ == "__main__":
    main()