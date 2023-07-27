#python3 -m uvicorn main:app --reload --port 7000
#wajib download nltk.punkt

from fastapi import FastAPI, Form
import re
import json
import requests
import pandas as pd
from bertopic import BERTopic
import emoji
import numpy
from sklearn.datasets import fetch_20newsgroups
import openai
from bertopic.representation import KeyBERTInspired
# from bertopic.representation import MaximalMarginalRelevance, OpenAI, PartOfSpeech
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from os import listdir
import matplotlib.pyplot as plt
import os

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SCRAPPING DATA REVIEW PRODUCT
@app.post("/analisa/")
async def analisa(named: Annotated[str, Form()], linked: Annotated[str, Form()]):
    df = pd.DataFrame(columns=['comments', 'star'])
    url = linked
    r = re.search(r'i\.(\d+)\.(\d+)', url)
    shop_id, item_id = r[1], r[2]
    ratings_url = 'https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0'

    offset = 0
    while True:
        data = requests.get(ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)).json()

        i = 1
        m = 0
        #print(data)
        try:
            for i, rating in enumerate(data['data']['ratings'], 1):
                #print(rating)
                star = rating['rating_star']
                s = rating['comment']
                s = s.strip()
                s = " ".join(s.split())
                s = emoji.replace_emoji(s, '')

                if s != "" and s != " " :
                    df.loc[len(df)] = [s, star]
                    m = m + 1

                #if i % 20:
                #    break

            offset += 20
        except Exception as e:
            print(e)
            break

    new_df = df.dropna()
    new_df['comments'] = new_df['comments'].replace({';': ''}, regex=True)
    new_df = new_df.dropna()
    new_df = new_df.reset_index(drop=True)
    new_df['comments'] = new_df['comments'].replace({'\"': ''}, regex=True)
    new_df.to_csv('csvfile/'+named+'.csv', index=False)
    # print(new_df.head())
    return {"status": "berhasil"}

@app.post("/historian/")
async def analisa(filename: Annotated[str, Form()]):
    #name = filename+'.csv'
    if os.path.isfile("imgfile/"+filename+".png"):
        os.remove("imgfile/"+filename+".png")

    if os.path.isfile("imgfile/topic_"+filename+".png"):
        os.remove("imgfile/topic_"+filename+".png")

    if os.path.isfile("imgfile/topic_a"+filename+".png"):
        os.remove("imgfile/topic_a"+filename+".png")

    # READING CSV FILE
    nltk.download('punkt')
    df = pd.read_csv('csvfile/'+filename)

    # PRINT RAW DATA
    df['new_comments'] = df['comments']
    # Print to json
    res = df['new_comments'].to_json(orient="records")
    parsed = json.loads(res)
    json_newcomments = parsed

    # Sentence Splitting to determine sentiment
    sentences = []
    for row in df.itertuples():
        for sentence in sent_tokenize(row[1]):
            sentences.append((row[1], sentence))
    df = pd.DataFrame(sentences, columns = ['comments','new_comments'])
    # Print to json
    res = df['new_comments'].to_json(orient='records')
    parsed = json.loads(res)
    json_splitting = parsed

    df[["neutral", "positive", "negative"]] = 0.0

    # Casefolding dan replace tanda baca
    df['new_comments'] = df['new_comments'].apply(str.lower)
    df['new_comments'] = df['new_comments'].replace({",": ""}, regex=True)
    df['new_comments'] = df['new_comments'].replace({":": " "}, regex=True)
    df['new_comments'] = df['new_comments'].replace({"\.": ""}, regex=True)
    df['new_comments'] = df['new_comments'].replace({"\?": ""}, regex=True)
    df['new_comments'] = df['new_comments'].replace({"()": ""}, regex=True)
    df['new_comments'] = df['new_comments'].replace({"²": ""}, regex=True)
    # Additional of Data Cleaning
    def vc_remove_punctuation(text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")
    df['new_comments'] = df['new_comments'].apply(vc_remove_punctuation)

    def vc_remove_number(text):
        return re.sub(r"\d+", "", text)
    df['new_comments'] = df['new_comments'].apply(vc_remove_number)
    # Print to json
    res = df['new_comments'].to_json(orient="records")
    parsed = json.loads(res)
    json_casefolding = parsed

    # Remove Punctuation
    df['new_comments'] = df['new_comments'].str.replace(r'[^\w\s]+', '')
    # Print to json
    res = df['new_comments'].to_json(orient="records")
    parsed = json.loads(res)
    json_punctuation = parsed

    # Tokenized and Stopword Removal
    df['tokenized'] = df.apply(lambda row: word_tokenize(row['new_comments']), axis=1)
    # Print to json
    res = df['tokenized'].to_json(orient="records")
    parsed = json.loads(res)
    json_tokenized = parsed

    _stop_words_1 = [ 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'akulah', 'amat', 'amatlah', 'andalah',
                    'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'ataukah',
                    'ataupun', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahwasanya', 'bakal',
                    'bakalan', 'balik', 'banget', 'bapak', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah',
                    'begitupun', 'belakang', 'belakangan', 'belum', 'belumlah', 'benarkah', 'benarlah', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa',
                    'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal', 'berdatangan', 'berikan', 'berikutnya', 'berjumlah', 'berkali-kali',
                    'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan', 'bermacam',
                    'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya',
                    'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasanya', 'bilakah', 'bisakah',
                    'bolehkah', 'bolehlah', 'bukankah', 'bukanlah', 'bukannya', 'bung', 'caranya', 'cukup', 'cukupkah', 'cukuplah', 'dan', 'dapat',
                    'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya',
                    'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan',
                    'digunakan', 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya',
                    'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat',
                    'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'wkwk',
                    'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan',
                    'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan',
                    'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan',
                    'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya',
                    'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'entahlah', 'gunakan', 'hal', 'hampir', 'hanya',
                    'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 'ok',
                    'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu',
                    'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas',
                    'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun',
                    'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata',
                    'katakan', 'katakanlah', 'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan',
                    'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian',
                    'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita',
                    'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat',
                    'lima', 'luar', 'macam', 'maka', 'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa',
                    'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'ttp',
                    'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan',
                    'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan',
                    'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan', 'menanya',
                    'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa',
                    'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya',
                    'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab', 'skrg',
                    'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut',
                    'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini',
                    'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah',
                    'naik', 'namun', 'nanti', 'nantinya', 'nya', 'nyaris', 'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling',
                    'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah',
                    'persoalan', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya',
                    'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai', 'sampai-sampai', 'sampaikan',
                    'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian',
                    'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa',
                    'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian',
                    'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak',
                    'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika', 'bsd',
                    'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya',
                    'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'sbg', 'ken',
                    'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuaa', 'semuanya', 'semula', 'sendiri', 'sendirian', 'lg',
                    'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'baguss', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya',
                    'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya',
                    'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi',
                    'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'udah', 'sudahkah', 'sudahlah', 'wkwkwk',
                    'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'cepet',
                    'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir',
                    'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'uu',
                    'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju',
                    'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'sdh', 'gpp', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk',
                    'turut', 'tutur', 'tuturnya', 'gitu', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'hehehe',
                    'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'ya', 'yaitu', 'yakin', 'yakni', 'yang', 'yg', 'dg', 'dr', 'thx', 'u', 'no',
                    'cpt', 'tp', 'dtng', 'dgn', 'jg', 'tgl', 'sampe', 'nyampe', 'kak', 'mantaaaap', 'blm', 'sukak', 'trims', 'tq', 'aja', 'kw', 'dh', 'udh', 'deh', 'bgt',
                    'dng', 'nich', 'jd', 'saaampaaai', 'bagusteeriiiiimaaa', 'nih', 'in', 'pus', 'eadaan', 'yey', 'huhu', 'selamattt', 'aga', 'si', 'cobain', 'a', 'd', 'sa']

    _stop_words_1 = set(_stop_words_1)
    df['stopwords'] = df['new_comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (_stop_words_1)]))

    # Print to json
    res = df['stopwords'].to_json(orient="records")
    parsed = json.loads(res)
    json_stopwords = parsed

    texts=df['stopwords']
    # print(texts.head())

    # topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    # topics, probs = topic_model.fit_transform(texts)
    # df2 = pd.DataFrame(topic_model.get_document_info(texts))

    topic_model = BERTopic(language="indonesian", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    df2 = pd.DataFrame(topic_model.get_document_info(texts))

    conditions = [
        (df2['Probability'] > 0) & (df2['Probability'] < 0.4),
        (df2['Probability'] >= 0.4) & (df2['Probability'] < 0.7),
        (df2['Probability'] >= 0.7)
    ]
    values = ['Negatif', 'Netral', 'Positif']
    df2['Sentiment'] = numpy.select(conditions, values)
    #print(df2.head())

    fig = topic_model.visualize_barchart(top_n_topics = 8, n_words = 10)
    fig.write_html("htmlfile/"+filename+".html")

    # print(topic_model.get_topic_info())
    # print(topic_model.get_topic(0))
    # print(topics)
    # print(probs)

    temporer = topic_model.get_topic(0)
    df_temporer=pd.DataFrame(temporer, columns=["x", "y"])
    df_temporer["hsl"] = df_temporer["y"] * 1000

    set_horizontal(filename, df_temporer['x'], df_temporer['hsl'])

    # Print to json
    res = df_temporer.to_json(orient="records")
    parsed = json.loads(res)
    json_topic_probability = parsed

    # Print to json
    res = df2.to_json(orient="records")
    parsed = json.loads(res)
    json_model = parsed

    tmp = topic_model.get_topic_info();

    # set_horizontal_a(filename, tmp['Name'], tmp['Count'])

    # Print to json
    res = tmp.to_json(orient="records")
    parsed = json.loads(res)
    json_topic_info = parsed

    df_merge = pd.merge(df,df2, how='inner', left_on = 'comments', right_on = 'Document')
    df_merge = df_merge.drop('Document', axis=1)
    # print(df_merge.head())

    # Print to json
    res = df_merge.to_json(orient="records")
    parsed = json.loads(res)
    json_merge = parsed

    netral = 0
    positive = 0
    negative = 0

    for index, row in df2.iterrows():
        if row['Probability'] >= 0 and row['Probability'] < 0.4:
            negative = negative + 1
        elif row['Probability'] >= 0.4 and row['Probability'] < 0.7:
            netral = netral + 1
        else :
            positive = positive + 1

    label = ['Negatif', 'Netral', 'Positif']
    jumlah = [negative, netral, positive]
    set_vertical(filename, label, jumlah)

    df_topic_sentiment = df_merge.groupby('Topic').agg({'neutral': 'mean', 'positive': 'mean', 'negative': 'mean'})
    df_topic_sentiment = df_topic_sentiment.reset_index()
    # print(df_topic_sentiment)

    # Print to json
    res = df_topic_sentiment.to_json(orient="records")
    parsed = json.loads(res)
    json_topic_sentiment = parsed

    freq = topic_model.get_topic_info()
    df_freq = pd.DataFrame(freq)
    df_new = pd.merge (df_freq, df_topic_sentiment, how = 'inner', on = 'Topic' )
    # print(df_new)

    # Print to json
    res = df_new.to_json(orient="records")
    parsed = json.loads(res)
    json_freq = parsed

    # add new column with the highest score
    score_cols = ['neutral', 'positive', 'negative']
    df_new['highest_score'] = df_new[score_cols].max(axis=1)

    # Print to json
    res = df_new.to_json(orient="records")
    parsed = json.loads(res)
    json_score= parsed

    # define function to calculate sentiment label
    final = ''
    def get_sentiment(row):
        if row['positive'] == row['highest_score']:
            final = 'positive'
        elif row['negative'] == row['highest_score']:
            final = 'negative'
        else:
            final = 'neutral'

    # apply function to each row and create new column
    df_new['sentiment'] = df_new.apply(get_sentiment, axis=1)
    # print(df_new)

    # Print to json
    res = df_new.to_json(orient="records")
    parsed = json.loads(res)
    json_sentiment= parsed

    # Label the topics by using one of the other topic representations, like KeyBERTInspired
    # df_label = topic_model.set_topic_labels({1: "Good Quality", 2: "Easy To Use", 3: "Low Watt"})
    # KeyBERT
    keybert_model = KeyBERTInspired()
    # Representation Model
    representation_model = {"KeyBERT" : keybert_model}
    keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
    df_label = topic_model.set_topic_labels(keybert_topic_labels)

    # Print to json
    res = df_label.to_json(orient="records")
    parsed = json.loads(res)
    json_topic_label = parsed

    tmp = topic_model.get_topic_info();

    return {
        "status": "berhasil",
        "json_newcomments" : json_newcomments,
        "json_splitting" : json_splitting,
        "json_casefolding": json_casefolding,
        "json_punctuation": json_punctuation,
        "json_tokenized": json_tokenized,
        "json_stopwords": json_stopwords,
        "json_model": json_model,
        "json_topic_info": json_topic_info,
        "json_topic_probability": json_topic_probability,
        "json_topic_label" : json_topic_label,
        "netral":netral,
        "positive":positive,
        "negative":negative
        # "json_merge": json_merge,
        # "json_topic_sentiment": json_topic_sentiment,
        # "json_freq": json_freq,
        # "json_score": json_score,
        # "json_sentiment": json_sentiment,
        # "final": final
    }

def set_horizontal(filename, name, count):
    fig1 = plt.figure("Figure 2")
    plt.barh(name, count)
    plt.ylabel("Topic")

    plt.xlabel("Count")
    plt.title("Data Topic Info - Probability Score")
    plt.savefig('imgfile/topic_'+filename+'.png',dpi=400)

# def set_horizontal_a(filename, name, count):
#     fig1 = plt.figure("Figure 3")
#     plt.barh(name, count)
#     plt.ylabel("Topic")

#     plt.xlabel("Count")
#     plt.title("Data Topic Info")
#     plt.savefig('imgfile/topic_a'+filename+'.png',dpi=400)

@app.post("/listcsv/")
async def listcsv(posisi: Annotated[str, Form()]):
    filenames = find_csv_filenames("csvfile")
    json_filename = "["
    for name in filenames:
        json_filename = json_filename + '{"item":"'+ name + '"},'
        #print(name)
    json_filename = json_filename.rstrip(json_filename[-1])
    json_filename = json_filename + "]"
    json_filename = json.loads(json_filename)
    # print(json_filename)
    return {
        "status" : "berhasil",
        "filename" : json_filename
    }

@app.post("/hapusan/")
async def analisa(filename: Annotated[str, Form()]):
    #name = filename+'.csv'
    if os.path.isfile("imgfile/"+filename+".png"):
        os.remove("imgfile/"+filename+".png")

    if os.path.isfile("htmlfile/"+filename+".html"):
        os.remove("htmlfile/"+filename+".html")

    if os.path.isfile("csvfile/"+filename):
        os.remove("csvfile/"+filename)

    return {
        "status" : "berhasil",
        "filename": filename
    }

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def set_vertical(filename, label, jumlah):
    fig1 = plt.figure("Figure 1")
    plt.bar(label, jumlah, color = ['red', 'yellow', 'green'])
    plt.suptitle('Bar Chart Sentiment Data')
    plt.savefig('imgfile/'+filename+'.png',dpi=400)

@app.get("/")
def home():
    return {
        "Hello": "World",
        "Nasi": "Menanak",
        "Jkn" : {
            "Topic Modeling": "BERTopic"
        }
    }

@app.get("/ope")
def home():
    return {"Hello": "World"}