"""Try documented auth formats: connect.params.auth.token and variants."""
import asyncio
import json
import hashlib
import hmac
import uuid
import websockets

def load_config():
    import os
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    with open(config_path) as f:
        return json.load(f)

async def try_auth(name, make_payload):
    config = load_config()
    token = config["gateway"]["auth"]["token"]
    port = config["gateway"]["port"]
    url = f"ws://127.0.0.1:{port}"
    
    try:
        async with websockets.connect(url, open_timeout=3) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            challenge = json.loads(raw)
            nonce = challenge.get("payload", {}).get("nonce", "")
            
            payload = make_payload(token, nonce)
            print(f"[{name}]")
            print(f"  SEND: {json.dumps(payload)[:200]}")
            await ws.send(json.dumps(payload))
            
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(resp)
            event = data.get("event", data.get("type", "?"))
            print(f"  RECV: [{event}] {json.dumps(data)[:300]}")
            
            if "error" not in event.lower() and "fail" not in event.lower():
                # Try to read another message
                try:
                    resp2 = await asyncio.wait_for(ws.recv(), timeout=3)
                    data2 = json.loads(resp2)
                    print(f"  RECV2: {json.dumps(data2)[:200]}")
                except:
                    pass
                print(f"  [SUCCESS!]")
                return True
    except websockets.exceptions.ConnectionClosed as e:
        print(f"  CLOSED: code={e.code} reason={e.reason}")
    except asyncio.TimeoutError:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")
    return False

async def main():
    config = load_config()
    token = config["gateway"]["auth"]["token"]
    
    formats = [
        # From docs: connect.params.auth.token
        ("connect + params.auth.token", lambda t, n: {
            "type": "connect",
            "params": {
                "auth": {"token": t},
                "nonce": n,
                "device": {"id": "throng4-bridge", "type": "node", "name": "Throng4 Python Bridge"}
            }
        }),
        
        # Variation: HMAC signature of nonce 
        ("connect + signature", lambda t, n: {
            "type": "connect",
            "params": {
                "auth": {"token": t},
                "nonce": n,
                "signature": hmac.new(t.encode(), n.encode(), hashlib.sha256).hexdigest(),
                "device": {"id": "throng4-bridge", "type": "node"}
            }
        }),
        
        # Variation: simpler flat
        ("connect flat token+nonce", lambda t, n: {
            "type": "connect",
            "token": t,
            "nonce": n
        }),
        
        # From docs search: event-based
        ("event connect.response", lambda t, n: {
            "type": "event",
            "event": "connect.response", 
            "payload": {
                "auth": {"token": t},
                "nonce": n
            }
        }),
        
        # Try passing token as subprotocol in URL
        ("connect with id only", lambda t, n: {
            "type": "connect",
            "id": str(uuid.uuid4()),
            "params": {
                "auth": {"token": t},
                "nonce": n
            }
        }),

        # Maybe it wants just the "connect" event name matching the challenge event pattern 
        ("event connect + auth", lambda t, n: {
            "event": "connect",
            "payload": {
                "auth": {"token": t},
                "nonce": n,
                "device": {"id": "throng4-bridge"}
            }
        }),
    ]
    
    for name, fn in formats:
        success = await try_auth(name, fn)
        if success:
            print(f"\n*** FOUND WORKING FORMAT: {name} ***")
            break
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
