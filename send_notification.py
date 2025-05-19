import requests

def send_ntfy_message(message, topic="test", server="ntfy.cmxu.io"):
    """
    Send a message to a ntfy server.
    
    Args:
        message (str): The message to send
        topic (str): The topic/channel name, defaults to 'test'
        server (str): The ntfy server address, defaults to 'ntfy.cmxu.io'
    
    Returns:
        requests.Response: The response from the server
    """
    url = f"https://{server}/{topic}"
    response = requests.post(
        url,
        data=message.encode(encoding='utf-8')
    )
    return response

# Example usage
if __name__ == "__main__":
    send_ntfy_message("Hello from Python!") 