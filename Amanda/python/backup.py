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
import string
from nltk.tokenize import word_tokenize
import nltk
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analisa/")
async def analisa(linked: Annotated[str, Form()]):
    df = pd.DataFrame(columns=['comments'])
    url = linked
    r = re.search(r'i\.(\d+)\.(\d+)', url)
    shop_id, item_id = r[1], r[2]
    ratings_url = 'https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0'

    offset = 0
    while True:
        data = requests.get(ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)).json()

        # uncomment this to print all data:
        # print(json.dumps(data, indent=4))

        i = 1
        m = 0
        #print(data)
        try:
            for i, rating in enumerate(data['data']['ratings'], 1):
                #print(rating['author_username'])
                s = rating['comment']
                s = s.strip()
                s = " ".join(s.split())
                s = emoji.replace_emoji(s, '')

                if s != "" and s != " " :
                    df.loc[len(df)] = [s]
                    m = m + 1
                    #print(s)
                    #print('-' * 80)

                if i % 20:
                    break

            offset += 20
        except Exception as e:
            print(e)
            break

    new_df = df.dropna()
    new_df['comments'] = new_df['comments'].replace({';': ''}, regex=True)
    new_df = df.dropna()
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.applymap(lambda x: x.replace('\"', ''))
    new_df.to_csv('datamentah2.csv', index=False)
    # print(new_df.head())
    return {"username": linked}

@app.get("/testing")
def ngetest():
    # docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    # print(docs)
    # df = pd.DataFrame(columns=['comments'])
    # url = 'https://shopee.co.id/Premium-Brill-Eighty-eight-Flannel-Shirt-077-i.32031549.1991571675'
    # r = re.search(r'i\.(\d+)\.(\d+)', url)
    # shop_id, item_id = r[1], r[2]
    # ratings_url = 'https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0'
    #
    # offset = 0
    # while True:
    #     data = requests.get(ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)).json()
    #
    #     # uncomment this to print all data:
    #     # print(json.dumps(data, indent=4))
    #
    #     i = 1
    #     m = 0
    #     #print(data)
    #     try:
    #         for i, rating in enumerate(data['data']['ratings'], 1):
    #             #print(rating['author_username'])
    #             s = rating['comment']
    #             s = s.strip()
    #             s = " ".join(s.split())
    #             s = emoji.replace_emoji(s, '')
    #
    #             if s != "" and s != " " :
    #                 df.loc[len(df)] = [s]
    #                 m = m + 1
    #                 #print(s)
    #                 #print('-' * 80)
    #
    #             if i % 20:
    #                 break
    #
    #         offset += 20
    #     except Exception as e:
    #         print(e)
    #         break
    #
    # new_df = df.dropna()
    # new_df['comments'] = new_df['comments'].replace({';': ''}, regex=True)
    # new_df = df.dropna()
    # new_df = new_df.reset_index(drop=True)
    # new_df = new_df.applymap(lambda x: x.replace('\"', ''))
    # new_df.to_csv('datamentah2.csv', index=False)
    # print(new_df.head())

    nltk.download('punkt')
    df = pd.read_csv("datamentah2.csv")
    df[["neutral", "positive", "negative"]] = 0.0

    # Casefolding dan replace tanda baca
    df['comments'] = df['comments'].apply(str.lower)
    df['comments'] = df['comments'].replace({",": ""}, regex=True)
    df['comments'] = df['comments'].replace({":": " "}, regex=True)
    df['comments'] = df['comments'].replace({"\.": ""}, regex=True)
    df['comments'] = df['comments'].replace({"\?": ""}, regex=True)

    # Remove Punctuation
    df['comments'] = df['comments'].str.replace(r'[^\w\s]+', '')

    # Stopword Removal
    df['tokenized'] = df.apply(lambda row: word_tokenize(row['comments']), axis=1)

    _stop_words_1 = [ 'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya',
                    'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila',
                    'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya',
                    'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
                    'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah',
                    'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah',
                    'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah',
                    'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah',
                    'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan',
                    'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya',
                    'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa', 'biasanya',
                    'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya',
                    'bulan', 'bung', 'cara', 'caranya', 'cukup', 'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada',
                    'datang', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara',
                    'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan',
                    'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan',
                    'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan',
                    'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini',
                    'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan',
                    'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah',
                    'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan',
                    'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu',
                    'empat', 'enggak', 'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah',
                    'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat',
                    'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya',
                    'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika',
                    'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu',
                    'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke',
                    'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan', 'kelihatannya', 'kelima', 'keluar',
                    'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya',
                    'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kok', 'kurang', 'lagi',
                    'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka',
                    'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya',
                    'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya',
                    'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan',
                    'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan',
                    'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan',
                    'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan',
                    'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki',
                    'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya',
                    'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya',
                    'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa',
                    'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan',
                    'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nya',
                    'nyaris', 'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti',
                    'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan', 'pertama',
                    'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya',
                    'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai', 'sampai-sampai',
                    'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana',
                    'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum',
                    'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara',
                    'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera',
                    'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali',
                    'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekarang', 'sekecil', 'seketika', 'sekiranya',
                    'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama',
                    'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya',
                    'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua',
                    'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang',
                    'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta',
                    'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah',
                    'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya',
                    'setidaknya', 'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal',
                    'soalnya', 'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah',
                    'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas',
                    'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa',
                    'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi',
                    'terjadilah', 'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan',
                    'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba',
                    'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya',
                    'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai',
                    'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'ya', 'yaitu', 'yakin', 'yakni', 'yang']

    _stop_words_1 = set(_stop_words_1)
    df['stopwords'] = df['comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (_stop_words_1)]))

    texts=df['stopwords']
    print(texts.head())

    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    df2 = pd.DataFrame(topic_model.get_document_info(texts))
    print(df2.head())

    df_merge = pd.merge(df,df2, how='inner', left_on = 'comments', right_on = 'Document')
    df_merge = df_merge.drop('Document', axis=1)
    print(df_merge.head())

    df_topic_sentiment = df_merge.groupby('Topic').agg({'neutral': 'mean', 'positive': 'mean', 'negative': 'mean'})
    df_topic_sentiment = df_topic_sentiment.reset_index()
    print(df_topic_sentiment)
    #
    freq = topic_model.get_topic_info()
    df_freq = pd.DataFrame(freq)
    df_new = pd.merge (df_freq, df_topic_sentiment, how = 'inner', on = 'Topic' )
    print(df_new)

    # add new column with the highest score
    score_cols = ['neutral', 'positive', 'negative']
    df_new['highest_score'] = df_new[score_cols].max(axis=1)

    # define function to calculate sentiment label
    def get_sentiment(row):
        if row['positive'] == row['highest_score']:
            return 'positive'
        elif row['negative'] == row['highest_score']:
            return 'negative'
        else:
            return 'neutral'

    # apply function to each row and create new column
    df_new['sentiment'] = df_new.apply(get_sentiment, axis=1)

    print(df_new)

    return { "Jalan": "yes" }

@app.get("/")
def home():
    return {
        "Hello": "Kyara",
        "Nasi": "Menanak",
        "Jkn" : {
            "Abdi": "Wahyuda"
        }
    }

@app.get("/ope")
def home():
    return {"Hello": "World"}
