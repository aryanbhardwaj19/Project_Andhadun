import urllib.request
import os

def download_test_images():
    # List of 10 face image URLs (sample faces from public datasets/placeholders)
    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face.png",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/group.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/faces.png",
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/ec/Elizabeth_II_in_Berlin_2015_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Angela_Merck_2016.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/af/Narendra_Modi_Official_Portrait_2022.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/85/Elon_Musk_Royal_Society_%28crop1%29.jpg"
    ]
    
    target_dir = "edge-assist/tests/data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    for i, url in enumerate(urls):
        ext = url.split(".")[-1]
        if len(ext) > 4: ext = "jpg" # default for messy urls
        filename = f"face_{i+1}.{ext}"
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    download_test_images()
