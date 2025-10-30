import re
import requests
from bs4 import BeautifulSoup
import random
import time
from datetime import datetime, timedelta

class Scraper():
    def __init__(self):
        self.cookie = {"Cookie":"SINAGLOBAL=7973288796724.17.1664160723535; SCF=Ak36-mSiYJz2fFofSw7NNYDa9Zd0oqgXgNpbcBeZT-658IYW925SM3m7nG10PtUUwFMXbr2EdBzemCcfB2lG6MA.; ALF=1764159302; SUB=_2A25F-xIWDeRhGeNO6VUT9S_EyDWIHXVneSverDV8PUJbkNANLRPAkW1NTx_CED2KMNLUnI3y078bmxW77FZ8_e76; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5g955MZ9Aq.BJbDC2nE.J05JpX5KMhUgL.Fo-7eoMESK2Re0.2dJLoIpjLxK.L1K.L1hnLxKqL1-eL1h.LxKnL12eL1het; _s_tentry=-; Apache=5805802571324.574.1761568631920; ULV=1761568631943:6:6:4:5805802571324.574.1761568631920:1761568481895"}
        self.cookie_mobile = {"Cookie":"_T_WM=f083755324d8f59bd817020ccf769702; SCF=Ak36-mSiYJz2fFofSw7NNYDa9Zd0oqgXgNpbcBeZT-65x9yvXJHDN1WqiU5rAaDmXdmg-XB4Felo5Rx-LlldLu0.; SUB=_2A25F-xIWDeRhGeNO6VUT9S_EyDWIHXVneSverDV6PUJbktANLXbFkW1NTx_CEFAz-7q9zq8m3doAJYYOkOWij0FX; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5g955MZ9Aq.BJbDC2nE.J05JpX5KMhUgL.Fo-7eoMESK2Re0.2dJLoIpjLxK.L1K.L1hnLxKqL1-eL1h.LxKnL12eL1het; SSOLoginState=1761567302; ALF=1764159302"}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Pragma': 'no-cache',
            'Referer': 'https://weibo.com/'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.cookies.update(self.cookie)
    def random_delay(self, min_delay=2, max_delay=5):
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    def rotate_user_agent(self):
        user_agents = [
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 LBBROWSER",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (iPad; U; CPU OS 4_2_1 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/6533.18.5",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        ]
        return random.choice(user_agents)
    def is_anti_crawler(self, response):
        """检测是否触发了反爬虫"""
        anti_crawler_indicators = [
            len(response.text) < 5000,  # 内容过少
            '验证' in response.text,     # 包含验证关键词
            'passport' in response.url,  # 被重定向到登录页
            'security' in response.text, # 安全验证
            'access denied' in response.text.lower(),
        ]
        return any(anti_crawler_indicators)
    def convert_to_mobile_url(self, url):
        """将微博PC链接转换为移动端链接"""
        # 匹配 weibo.com 并替换为 weibo.cn
        mobile_url = re.sub(r'https?://weibo\.com', 'https://weibo.cn', url)
        return mobile_url
    def keyword_query(self,keyword,times):
        '''通过关键字搜索得到的文章url'''
        init_url = 'https://s.weibo.com/weibo'
        url = init_url + '/' + keyword
        self.session.cookies.update(self.cookie)
        res = self.session.get(url,timeout=10)
        urls = []
        ct = 0
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            
            from_divs = soup.find_all('div', class_='from')
    
            for div in from_divs:
                a_tag = div.find('a')
                if a_tag and a_tag.has_attr('href'):
                    href_url = "https:" + a_tag['href']
                    urls.append(self.convert_to_mobile_url(href_url))
                    ct += 1
                if ct == times:
                    break
            
        self.random_delay()
        self.headers["User-Agent"] = self.rotate_user_agent()
        return urls
    def content_comment(self,url):
        '''获取url对应文章的相关信息'''
        page = 0
        self.session.cookies.update(self.cookie_mobile)
        comments_count = 0
        content = {
            'text':'',
            'img':[],
            'time':''
        }
        comments = []
        while(1):
            page += 1
            res = self.session.get(url + f"&page={page}",timeout=10)
            tmp_ct = 0
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                detail_divs = soup.find_all('div', class_='c')
                for i, div in enumerate(detail_divs, 1):
                    a = div.find('a')
                    if not a:
                        continue
                    name = a.get_text()
                    span = div.find('span', class_='ctt')
                    if not span:
                        continue
                    text = span.get_text(strip=True)
                    if not text:
                        continue
                    tmp_ct += 1
                    if tmp_ct == 1:
                        if content['text'] == '':
                            content['text'] = text[1:]
                            images = div.find_all('img',class_='ib')
                            for img in images:
                                img_src = img.get('src')
                                content['img'].append(img_src)
                            content_time = div.find('span',class_='ct').get_text()[:19]
                            if '今天' in content_time:
                                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                content_time = current_time[:11] + content_time[3:8] + ":00"
                            elif '分钟前' in content_time:
                                minite = int(content_time.split('分')[0])
                                now = datetime.now()
                                content_time = (now - timedelta(minutes=minite)).strftime("%Y-%m-%d %H:%M:%S")
                            elif '月' in content_time:
                                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                if content_time[9] == ':':
                                    content_time = current_time[:5] + content_time[:2] + current_time[7] + content_time[3:5] + current_time[10] + content_time[7:12] + current_time[16:]
                                elif content_time[7] == ':':
                                    content_time = current_time[:5] + '0' + content_time[0] + current_time[7] + '0' + content_time[2] + current_time[10] + content_time[5:10] + current_time[16:]
                                elif content_time[1] == '月':
                                    content_time = current_time[:5] + '0' + content_time[0] + current_time[7] + content_time[2:4] + current_time[10] + content_time[6:11] + current_time[16:]
                                else :
                                    content_time = current_time[:5] + content_time[:2] + current_time[7] + '0' + content_time[3] + current_time[10] + content_time[6:11] + current_time[16:]
                            content['time'] = content_time
                        continue
                    comments_count += 1
                    is_reply = text[0] == '@'
                    rep_time = div.find('span',class_='ct').get_text()[:19]
                    tmp_cmt = {
                        'text':text,
                        'name':name,
                        'is_reply':is_reply,
                        'reply2':'',
                        'time':rep_time,
                    }
                    if is_reply:
                        a_ = span.find('a')
                        reply2 = a_.get_text()[1:]
                        tmp_cmt['reply2'] = reply2
                    comments.append(tmp_cmt)
                    if comments_count == 10:
                        break
            else:
                break
            if tmp_ct == 1:
                return content,comments
    def query(self,keyword,times = 3):
        '''外部通过关键词查询以及设置查询的数量限制'''
        urls = self.keyword_query(keyword, times)
        contents = []
        comments = []
        ct = 0
        for url in urls:
            c1,c2 = self.content_comment(url)
            contents.append(c1)
            comments.append(c2)
            ct += 1
            if ct == times:
                break
        return contents,comments
    def downloadimg(self,img_url,path='image.jpg'):
        '''下载图片到指定路径'''
        response = requests.get(img_url, headers=self.headers, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
    def hot_search(self):
        url = 'https://weibo.com/ajax/side/hotSearch'
        self.session.cookies.update(self.cookie)
        res = self.session.get(url,timeout=10)
        if res.status_code == 200:
            data = res.json()
            texts = []
            nums = []
            if 'data' in data and 'realtime' in data['data']:
                for item in data['data']['realtime']:
                    word = item.get('word', '')
                    note = item.get('note', '')
                    num = int(item.get('num', ''))
                    text = note if note else word
                    if text:
                        texts.append(text)
                        nums.append(num)
            
            return texts,nums
        else:
            return []
