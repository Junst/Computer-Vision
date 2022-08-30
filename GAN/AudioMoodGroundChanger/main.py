from pytube import YouTube
# import ssl
# ssl._create_default_https_context = ssl._create_stdlib_context

DOWNLOAD_FOLDER = "/videos/" # C 드라이브 바로 안에 있음

#가져올 링크 넣기
url = "https://www.youtube.com/watch?v=EAgACSowE5k" # 아라시 Love so sweet
yt = YouTube(url)

# 정보 가져오기
print("title : ", yt.title)
print("length : ", yt.length)
print("author : ", yt.author)
print("publish_date : ", yt.publish_date)
print("views : ", yt.views)
print("keywords : ", yt.keywords)
print("description : ", yt.description)
print("thumbnail_url : ", yt.thumbnail_url)

# 비디오 다운로드
from pytube import Playlist
p = Playlist('https://www.youtube.com/watch?v=cJZCQdAYRFM&list=PLKRZTF1Q1uwaeTOXQ3BwQLQJJ32c3wlUW')
for video in p.videos:
    video.streams.first().download(DOWNLOAD_FOLDER)
    print("다운로드 완료")
