# backend/test.py
import sys
import json
import io
from scrap_data import Scraper
from rumor_detect_model import RumorDetector
import random

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    scraper = Scraper()
    detector = RumorDetector()
    detector.load('./model/model.pth')
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        times = random.randint(1,3)
        contents,comments = scraper.query(keyword,times)
        'id', 'text', 'source', 'probability', 'time','result','dealed','image'
        results = []
        for i in range(len(contents)):
            content = contents[i]
            comment = comments[i]
            if len(comment) > 10:
                comment = comment[:10]
            pred_res = detector.predict(content,comment)
            p = pred_res['confidence']
            result = {
                'id':i+1,
                'text':content['text'],
                'source':'新浪微博',
                'probability':p,
                'time':content['time'],
                'result':False,
                'dealed':False,
                'image':content['img']
            }
            results.append(result)
        
        
        # 输出结果（会被Node.js捕获）
        response = {
            "success": True,
            "keyword": keyword,
            "data": results,
            "count": len(results)
        }
        print(json.dumps(response, ensure_ascii=False))
            
    else:
        error_response = {
            "success": False,
            "error": "未提供关键词参数"
        }
        print(json.dumps(error_response, ensure_ascii=False))

if __name__ == "__main__":
    main()