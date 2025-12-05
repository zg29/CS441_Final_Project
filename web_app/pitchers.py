import requests
from bs4 import BeautifulSoup
from pybaseball import playerid_lookup
import requests
from bs4 import BeautifulSoup

def get_pitcher_photo_url(name):
    last, first = name.split()
    data = playerid_lookup(last, first)
    player_id = data.key_bbref.iloc[0]
    print("PLAYER IDDDDDDDDDD", player_id)
    first_letter = str(player_id)[0]
    url = f"https://www.baseball-reference.com/players/{first_letter}/{player_id}.shtml"
    print(f"URL: {url}")

    response = requests.get(url)
    if response.status_code == 200:
        page_content = response.text
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        exit()

    soup = BeautifulSoup(page_content, 'html.parser')

    media_div = soup.find('div', class_='media-item multiple')

    if media_div:
        img_tags = media_div.find_all('img')

        headshot_urls = [img['src'] for img in img_tags]
        print("Headshot URLs:")
        for url in headshot_urls:
            return(url)
    else:
        print("Could not find the player's headshot images on the page.")