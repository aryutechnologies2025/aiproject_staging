from selenium.webdriver.common.by import By
import time
import pickle
import hashlib
import undetected_chromedriver as uc

COOKIE_PATH = "/home/aryu_user/Arun/aiproject_staging/app/modules/resume_builder/linkedin/linkedin_cookies.pkl"


def create_driver(headless: bool = False, profile_dir: str = None):
    options = uc.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")

    if profile_dir:
        options.add_argument(f"--user-data-dir={profile_dir}")

    driver = uc.Chrome(options=options)
    return driver


def load_cookies(driver):
    driver.get("https://www.linkedin.com/")
    cookies = pickle.load(open(COOKIE_PATH, "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)


def scroll_page(driver):
    for _ in range(6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)


def extract_profile_text(url: str):
    driver = create_driver()
    load_cookies(driver)

    driver.get(url)
    time.sleep(5)

    scroll_page(driver)

    text = driver.find_element(By.TAG_NAME, "body").text

    driver.quit()

    return text


def generate_hash(text: str):
    return hashlib.md5(text.encode()).hexdigest()