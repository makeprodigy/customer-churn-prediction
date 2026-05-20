import time
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        # Launch a headless Chromium browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        app_url = "https://customer-churn-prediction-m2-sem4.streamlit.app/"

        print(f"Navigating to {app_url}...")
        page.goto(app_url, timeout=60000)

        # Wait for the page to fully load and WebSocket connection to establish
        # Streamlit needs ~15s for the websocket handshake to count as "active"
        print("Waiting for app to fully load and WebSocket to connect...")
        time.sleep(20)

        # Check if the page title loaded correctly
        title = page.title()
        print(f"Page title: {title}")

        print("✅ App visited successfully. Hibernation timer reset!")
        browser.close()


if __name__ == "__main__":
    run()
