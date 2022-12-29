from pytube import YouTube
import os
import sys
import pandas as pd
from utils.utils import is_url

def download_youtube(vid_url, path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    extension = "mp4"
    yt = YouTube(vid_url)
    if f"{yt.title}.{extension}" in os.listdir(path_dir):
        print(f"The url {vid_url} has already downloaded to: {yt.title}")
    else:
        yt = yt.streams.filter(progressive=True, file_extension=extension).order_by('resolution').desc().first()
        yt.download(path_dir)


def get_video_name(vid_url):
    yt = YouTube(vid_url)
    return yt.title

# video url or path to csv file with list of urls in columns with the name "urls"
origin= sys.argv[1]
# path to directory where you want to save the video
path = sys.argv[2]

add_names_flag = False
if is_url(origin):
    url = origin
    download_youtube(url, path)
else:
    df = pd.read_csv(origin)
    urls = df["url"].values
    print(urls[0])
    print(f"number of urls: {len(urls)}")
    for url in urls:
        try:
            download_youtube(url, path)
            print(f"{url} has been downloaded successfully ")
        except:
            print(f"!! problem with {url} downloading !!")

if add_names_flag:
    # add video name to the videos dictionary
    video_names = []
    df = pd.read_csv(origin)
    urls = df["url"].values
    for url in urls:
        video_names.append(YouTube(url).title)
    df["video_name"] = video_names
    df.to_csv(origin)

exit()