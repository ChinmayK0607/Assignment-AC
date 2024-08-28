import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    # Create the checkpoints directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))

    # Open the destination file and write the content
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

if __name__ == "__main__":
    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
    destination = os.path.join("checkpoints", "depth_anything_v2_vits.pth")
    
    print(f"Downloading model from {url}")
    download_file(url, destination)
    print(f"Model downloaded and saved to {destination}")
