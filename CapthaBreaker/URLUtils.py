import requests

def isDownloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True

def saveImageFromURL(url, path, name):
    r = requests.get(url, allow_redirects=True)
    open(path + name, 'wb').write(r.content)

