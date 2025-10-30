# backend/test.py
import sys
import json
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import jieba.analyse
from scrap_data import Scraper
import re
def texts2keywords(titles):
    """专门针对微博热搜短标题的关键词提取"""
    stop_words = {
        '如何', '怎样', '什么', '为什么', '怎么办', '哪些', '哪个', 
        '哪里', '解析', '分析', '解读', '看法', '观点', '评论',
        '真的', '就是', '可以', '应该', '需要', '要求', '建议'
    }
    
    important_words = {
        '首位', '首次', '第一', '首度', '突破', '创新', '重大',
        '新', '全新', '最新', '首款', '首秀', '首播'
    }
    
    keywords_list = []
    
    for title in titles:
        clean_title = re.sub(r'#.*?#', '', title)
        clean_title = re.sub(r'[【】]', '', clean_title)
        
        try:
            keywords = jieba.analyse.textrank(
                clean_title, 
                topK=8,
                withWeight=False,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'a', 'eng')
            )
        except:
            keywords = []
        
        filtered_keywords = []
        for kw in keywords:
            if (kw in stop_words or 
                len(kw) <= 1 or 
                kw.isdigit() or
                kw in ['的', '了', '在', '是', '有', '和', '就']):
                continue
            
            if kw in important_words:
                filtered_keywords.append(kw)
                continue
                
            if 2 <= len(kw) <= 6:
                filtered_keywords.append(kw)
        
        filtered_keywords = list(dict.fromkeys(filtered_keywords))[:5]
        
        if len(filtered_keywords) < 2:
            try:
                tfidf_keywords = jieba.analyse.extract_tags(
                    clean_title, 
                    topK=5,
                    allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'a', 'eng')
                )
                for kw in tfidf_keywords:
                    if (kw not in stop_words and 
                        len(kw) >= 2 and 
                        kw not in filtered_keywords and
                        not kw.isdigit()):
                        filtered_keywords.append(kw)
                filtered_keywords = filtered_keywords[:5]
            except:
                pass
        
        if not filtered_keywords:
            words = jieba.lcut(clean_title)
            backup_keywords = [word for word in words 
                             if len(word) >= 2 and word not in stop_words]
            filtered_keywords = list(dict.fromkeys(backup_keywords))[:3]
        
        keywords_list.append(filtered_keywords)
    
    return keywords_list
def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    results = []
    keywords = []
    scraper = Scraper()
    texts,nums = scraper.hot_search()
    init_keywords = texts2keywords(texts)
    max_num = max(nums)
    min_num = min(nums)
    scale_size = max_num - min_num

    for i in range(len(texts)):
        keyword_list = init_keywords[i]
        if keyword_list:
            for keyword in keyword_list:
                k_n = {'name':keyword,'weight':1 + ((nums[i] - min_num) * 9) // scale_size}
                keywords.append(k_n)
                
    response = {
        "success": True,
        "keywords": keywords,
        "data": results,
        "count": len(results)
    }
    print(json.dumps(response, ensure_ascii=False))
if __name__ == "__main__":
    main()