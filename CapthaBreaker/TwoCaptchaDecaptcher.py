import requests
import time

class TwoCaptchaDecaptcher:

    """
    This class contains method to solve a Captcha using 2Captcha API.
    """

    TWO_CAPTCHA_POST_URL = 'http://2captcha.com/in.php'
    TWO_CAPTCHA_GET_URL = 'http://2captcha.com/res.php'

    def __init__(self, apiKey):
        self.TWO_CAPTCHA_API_KEY = apiKey;

    def get(self, params, **kwargs):
        params['key'] = self.TWO_CAPTCHA_API_KEY
        return requests.get(self.TWO_CAPTCHA_GET_URL, params, **kwargs)

    def post(self, params, **kwargs):
        params['key'] = self.TWO_CAPTCHA_API_KEY
        return requests.post(self.TWO_CAPTCHA_POST_URL, params, **kwargs)

    def sendImage(self, captchaImage):

        #Check if file or image properly
        if type(captchaImage) == str:
            captchaImageFile = open(captchaImage, 'rb')
            imageToSend = captchaImageFile.read()
        else:
            imageToSend = captchaImage


        #Request and get answer
        method = {'method':'post'}
        files={'file': ('captcha.jpg', imageToSend)}

        answer = self.post(method, files = files).text

        captchaID = answer
        if '|' in answer:
            _, captchaID = answer.split('|')
            return captchaID
        else:
            return captchaText

    def decaptcha(self, captchaImage):

        #Send image to API and get ID back
        captchaID = self.sendImage(captchaImage)

        #Make a request to API to get the captcha code back
        params = {'action':'get', 'id':captchaID}

        answer = 'NOT_READY'

        while 'NOT_READY' in answer:
            time.sleep(2)
            answer = self.get(params).text

        captchaText = answer
        if '|' in answer:
            _, captchaText = answer.split('|')
            return captchaText
        else:
            return captchaText







