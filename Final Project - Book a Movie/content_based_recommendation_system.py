##########################################################
# Content and Rate Based Hybrid Recommendation System
##########################################################

# Film Tavsiyesi
    # Keşifçi Veri Analizi
    # Seçilen 1 filme karşılık 5 film önerilmesi
    # Seçilen 2 filme karşılık 5 film önerilmesi

# Kitap Tavsiyesi
    # Keşifçi Veri Analizi
    # Seçilen 1 kitaba karşılık 5 kitap önerilmesi
    # Seçilen 2 kitaba karşılık 5 kitap önerilmesi

# Seçilen filme karşılık 5 kitap önerilmesi
# Seçilen kitaba karşılık  5 film önerilmesi


##########################################################
##################   FİLM   #############################
##########################################################

##########################################################
# Film Açıklamalarına Göre Tavsiye Geliştirme
##########################################################

import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


# Sonunda Read All ifadesi geçenlerin sonundan bu ifadeyi silmek için kullanacağımız fonksiyon.
def remove_read_all(text):
    if isinstance(text, str) and text.endswith('Read all'):
        return text[:-8].strip()  # 'Read all' ifadesini ve önceki boşluğu kaldır
    return text


# User Rating değişkeni 187K, 6M şeklinde kaydedildiğinden, bu değişkeni sayısal değerlere dönüştürmemiz gerekiyor.
def convert_to_int(rating):
    if 'K' in rating:
        return int(float(rating.replace('K', '')) * 1_000)
    elif 'M' in rating:
        return int(float(rating.replace('M', '')) * 1_000_000)
    else:
        return int(rating)


# Ağırlıkları ayarlamak için fonksiyon.
def weighted_sorting_score(dataframe, w1=48, w2=52):
    return (dataframe["Rating_Count_Scaled"] * w1 / 100 +
            dataframe["Rating"] * w2 / 100)


def clean_movies():
    # Verinin okunması
    df = pd.read_csv("../../Miuul-Bootcamp-Project/data/25k IMDb movie Dataset.csv")
    print("### Veriyi inceleyelim ###")
    print(df.head())
    print("### Veriyi boyutu ###")
    print(df.shape)

    # Verinin uygun formata getirilmesi
    print("### Boş değerleri inceleyelim ###")
    print(df.isnull().sum())

    print("### Sayısal değerleri inceleyelim ###")
    print(df.describe().T)

    # Overview değeri boş ve none olanları düşüreşim.
    # Veri seti none olarak kaydedilmiş açıklamalar içeriyordu.
    df = df.dropna(subset=['Overview'])
    df = df[df['Overview'] != 'none']

    df['Overview'] = df['Overview'].apply(remove_read_all)

    # Filmlerin Overview ve Plot Kyeword Sütunlarını Birleştir ve Description değişkenini oluştur.
    df['Plot Kyeword'] = df['Plot Kyeword'].apply(literal_eval)
    df['Description'] = df['Overview'] + ' ' + df['Plot Kyeword'].apply(lambda x: ' '.join(x))

    # Belirli bir harf sayısının altında olan açıklamaları veri setinden çıkartıyoruz.
    # Kelime sayısını hesaplayıp ve word_count değişkenine atıyoruz
    df['word_count'] = df['Description'].str.split().str.len()

    # Kelime eşiğini belirle ve veri setini filtrele
    threshold = 15
    df = df[df['word_count'] > threshold]

    # Rating ve User Rating değişkenlerini standarlaştırıyoruz.
    df["Rating"].dtype  # object
    df["User Rating"].dtype  # object

    # Değişken tipini düzenliyoruz.
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Dönüştürme işlemini uyguluyoruz
    df["User Rating"] = df["User Rating"].apply(convert_to_int)

    # User Rating'i 0 olanları düşürüyoruz.
    df = df[df["User Rating"] != 0]

    # Rating'i boş olanları çıkarıyoruz.
    df = df.dropna(subset=['Rating'])

    # User Rating değişkenini Rating'e uygun şekilde standartlaştırıyoruz.
    df["Rating_Count_Scaled"] = MinMaxScaler(feature_range=(1, 10)). \
        fit(df[["User Rating"]]). \
        transform(df[["User Rating"]])

    # Ve ağırlıklandırarak Weighted Rating değişkenini oluşturuyoruz.
    df["Weighted_Rating"] = weighted_sorting_score(df)

    # Tarih değerlerindeki negatif işaretleri kaldırıyoruz.
    df["year"].dtype
    df['year'] = df['year'].astype(str).str.replace('-', '')

    # Sütun adındaki yazım hatasını düzenliyoruz.
    df = df.rename(columns={'Generes': 'Genres'})

    # İhtiyacımız kalmayan değişkenleri kaldırıyoruz.
    columns_to_drop = ["Run Time", "Rating", "Overview", "Plot Kyeword", "User Rating", "Rating_Count_Scaled",
                       "word_count"]

    df = df.drop(columns=columns_to_drop)

    # Veri setimizi Weighted Rating değişkenine göre sıralandırıyoruz.
    df = df.sort_values("Weighted_Rating", ascending=False)

    # İndeksleri sıfırlıyoruz.
    df = df.reset_index(drop=True)

    df.to_excel('data/movies_duzenlendi.xlsx', index=False)

    return df


df = clean_movies()

df.head()
df.shape


##########################################################
# Seçilen filme karşılık 5 film önerisi
##########################################################

df = pd.read_excel('data/movies_duzenlendi.xlsx')

# tfidf nesnesini oluşturuyoruz.
tfidf = TfidfVectorizer(stop_words='english')

# Matrisi oluşturduk.
tfidf_matrix = tfidf.fit_transform(df['Description'])

tfidf_matrix.shape

# Satır sayısının doğruluğunu teyit ediyoruz.
df['movie title'].shape

tfidf.get_feature_names_out()

# Matrisin array formunu inceliyoruz.
tfidf_matrix.toarray()

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape

indices = pd.Series(df.index, index=df['movie title'])

# Bu bize çoktan aza doğru title'lar ve kaç kere bulunduklarını getirir.
indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='first')]
indices.index.value_counts()

# Öneri listesini saklayacak sözlük
recommendation_dict = {}

# Her film için benzer filmleri bul ve öneri listesine ekle
for movie_name in df['movie title']:
    movie_index = indices[movie_name]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    similarity_scores['Weighted_Rating'] = df['Weighted_Rating']

    # Skorları sıralama (seçilen filmi dahil etmemek için ilk satırı atlıyoruz)
    sorted_scores = similarity_scores.sort_values(by=['score', 'Weighted_Rating'], ascending=False).iloc[1:]

    # Önerilen filmlerin indekslerini alma
    movie_indices = sorted_scores.index

    # Önerilen film isimlerini listeye ekleme
    recommended_movies = []
    for idx in movie_indices:
        recommended_movie = df['movie title'].iloc[idx]
        if recommended_movie not in recommended_movies and recommended_movie != movie_name:
            recommended_movies.append(recommended_movie)
        if len(recommended_movies) == 5:
            break

    recommendation_dict[movie_name] = recommended_movies


# Sözlüğü bir veri çerçevesine dönüştür
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Film_title', 'Recommended Films'])

# Veri çerçevesini kontrol et
print(recommendation_df.head())

# Veri çerçevesini Excel dosyasına kaydet
recommendation_df.to_excel("data/movie_recommendations.xlsx", index=False)


##########################################################
# 2 Filme 5 Film Önerisi
##########################################################

df = pd.read_excel('data/movies_duzenlendi.xlsx')

# Weighted Rating'e göre sıraladığımız veri setimizin ilk 1000 filmini alıyoruz.
df = df.head(1000)

# TF-IDF vektörizer oluştur
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Cosine similarity matrisini oluştur
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Film adları ve indekslerini içeren pandas Series oluştur
indices = pd.Series(df.index, index=df['movie title'])

# Duplicate olmayanları tut
indices = indices[~indices.index.duplicated(keep='first')]

# Öneri listesini saklayacak sözlük
recommendation_dict = {}

# Her iki film için benzer filmleri bul ve öneri listesine ekle
for movie1 in df['movie title']:
    for movie2 in df['movie title']:
        if movie1 != movie2:
            # İlk filmin ve ikinci filmin indekslerini bul
            index1 = indices[movie1]
            index2 = indices[movie2]

            # Benzerlik skorlarını al
            scores1 = cosine_sim[index1]
            scores2 = cosine_sim[index2]

            # İki film arasındaki benzerlik skorlarını karşılaştır
            combined_scores = (scores1 + scores2) / 2

            # Kendini ve diğer seçilen filmi öneri olarak almayalım
            combined_scores[index1] = -1
            combined_scores[index2] = -1

            # Benzerlik skorlarını DataFrame'e dönüştür ve Weighted_Rating ile birleştir
            similarity_scores = pd.DataFrame({
                'score': combined_scores,
                'Weighted_Rating': df['Weighted_Rating']
            })

            # Skorları sıralama (seçilen filmi dahil etmemek için ilk satırı atlıyoruz)
            sorted_scores = similarity_scores.sort_values(by=['score', 'Weighted_Rating'], ascending=False)

            # En yüksek benzerlik skorlarına sahip 7 filmi bul (kendini ve diğer filmi çıkararak)
            top_7_indices = sorted_scores.index[:7]

            # Önerilen film isimlerini listeye ekleme
            recommendations = [df['movie title'].iloc[idx] for idx in top_7_indices if idx != index1 and idx != index2]

            # Sadece 5 öneri ile sınırlayalım
            recommendations = recommendations[:5]

            # İki film için öneri listesini dictionary'ye ekleyelim
            recommendation_dict[(movie1, movie2)] = recommendations

# Sözlüğü bir veri çerçevesine dönüştür
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Film Pair', 'Recommended Films'])

# Veri çerçevesini kontrol et
print(recommendation_df.head())

recommendation_df.shape

# Veri çerçevesini Excel dosyasına kaydet
recommendation_df.to_excel("data/movie_recommendations2.xlsx", index=False)





##########################################################
####################   KİTAP   ###########################
##########################################################


##########################################################
# Kitap Açıklamalarına Göre Tavsiye Geliştirme
##########################################################

# Ağırlıkları ayarlamak için fonksiyon.
def weighted_sorting_score(dataframe, w1=48, w2=52):
    return (dataframe["Rating_Count_Scaled"] * w1 / 100 +
            dataframe["average_rating"] * w2 / 100)

def clean_books():

    # Verinin okunması
    df = pd.read_csv("../../Miuul-Bootcamp-Project/data/data.csv")
    print("### Veriyi inceleyelim ###")
    print(df.head())
    print("### Veriyi boyutu ###")
    print(df.shape)

    print("### Boş değerleri inceleyelim ###")
    print(df.isnull().sum())

    print("### Sayısal değerleri inceleyelim ###")
    print(df.describe().T)

    # Verinin uygun formata getirilmesi

    # Bizim için kritik olan değişkenlerden boş değere sahip satırları düşürüyoruz.
    df = df.dropna(subset=['description', 'average_rating', 'thumbnail', 'categories', 'authors'])

    # Kelime sayısını hesaplayıp yeni bir sütun ekliyoruz
    df['word_count'] = df['description'].str.split().str.len()

    # Kelime eşiğini belirleyen filtrasyon
    threshold = 15
    df = df[df['word_count'] > threshold]

    # ratings_count değişkenini average_rating değerlerine uygun şekilde standartlaştırıyoruz.
    df["ratings_count"].dtype

    df["Rating_Count_Scaled"] = MinMaxScaler(feature_range=(1, 5)). \
        fit(df[["ratings_count"]]). \
        transform(df[["ratings_count"]])

    df.describe().T

    # Ve Weighted_Rating değişkenini oluşturuyoruz.
    df["Weighted_Rating"] = weighted_sorting_score(df)

    # İhtiyacımız kalmayan sütunları kaldırıyoruz.
    columns_to_drop = ["isbn13", "isbn10", "subtitle", "ratings_count", "average_rating", "Rating_Count_Scaled", "word_count"]

    df_dropped = df.drop(columns=columns_to_drop)

    # Veri setini Weighted_Rating değerlerine göre sıralıyoruz.
    df = df_dropped.sort_values("Weighted_Rating", ascending=False)

    # İndeksleri sıfırlıyoruz.
    df = df.reset_index(drop=True)

    print("### Sayısal değerleri işlemler sonrasında tekrar inceleyelim ###")
    print(df.describe().T)

    df.to_excel("data/books_duzenlendi.xlsx", index=False)

    return df

df = clean_books()

df.head()
df.shape  # (5792, 8)


##########################################################
# Seçilen kitaba karşılık 5 kitap önerisi
##########################################################

df = pd.read_excel("data/books_duzenlendi.xlsx")
df.head()

# tfidf nesnesini oluşturuyoruz.
# Yaygınca kullanılan ve ölçüm değeri taşımayan kelimeleri sil diyoruz.
tfidf = TfidfVectorizer(stop_words='english')

df[df['description'].isnull()]  # boş olan satırlar gelir.

# Matrisi oluşturduk.
tfidf_matrix = tfidf.fit_transform(df['description'])

tfidf_matrix.shape

# Satır sayısının doğruluğunu teyit ediyoruz.
df['title'].shape

# Sütunları inceliyoruz.
tfidf.get_feature_names_out()

# Matrisin array formunu inceliyoruz.
tfidf_matrix.toarray()

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape

indices = pd.Series(df.index, index=df['title'])

# Bu bize çoktan aza doğru title'lar ve kaç kere bulunduklarını getirir.
# Kitaplarda çoklama problemi ile karşılaştık.
indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='first')]
indices.index.value_counts()

# Öneri listesini saklayacak sözlük
recommendation_dict = {}

# Her kitap için benzer kitapları bul ve öneri listesine ekle
for book_name in df['title']:
    # Kitap indexini bulma
    book_index = df.index[df['title'] == book_name][0]

    # Benzerlik skorlarını dataframe'e dönüştürme
    similarity_scores = pd.DataFrame(cosine_sim[book_index], columns=["score"])

    # Weighted_Rating ile birleştirme
    similarity_scores['Weighted_Rating'] = df['Weighted_Rating']

    # Skorları sıralama (kendi kitabını dahil etmemek için ilk satırı atla)
    sorted_scores = similarity_scores.sort_values(by=['score', 'Weighted_Rating'], ascending=False).iloc[1:]

    # Önerilen kitapların indexlerini alma
    book_indices = sorted_scores.index

    # Önerilen kitap isimlerini listeye ekleme
    recommended_books = []
    for idx in book_indices:
        recommended_book = df['title'].iloc[idx]
        if recommended_book not in recommended_books and recommended_book != book_name:
            recommended_books.append(recommended_book)
        if len(recommended_books) == 5:
            break

    recommendation_dict[book_name] = recommended_books

# Sözlüğü bir veri çerçevesine dönüştür
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Book Name', 'Recommendations'])

# Veri çerçevesini kontrol et
print(recommendation_df.head())

# Veri çerçevesini Excel dosyasına kaydet
recommendation_df.to_excel("data/book_recommendations_duzenlendi.xlsx", index=False)
recommendation_df.to_csv("data/book_recommendations_duzenlendi.csv", index=False)



##########################################################
# 2 kitaba 5 kitap önerisi
##########################################################

df = pd.read_excel("data/books_duzenlendi.xlsx")
df.head()
# Weighted Rating'e göre sıraladığımız veri setimizin ilk 1000 kitabını alıyoruz.
df = df.head(1000)

# TF-IDF vektörizer oluştur
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Cosine similarity matrisini oluştur
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Kitap adları ve indekslerini içeren pandas Series oluştur
indices = pd.Series(df.index, index=df['title']).to_dict()

# Duplicate olmayanları tut
indices = {key: value for key, value in indices.items() if not pd.isnull(key)}

# Öneri listesini saklayacak sözlük
recommendation_dict = {}

# Her iki kitap için benzer kitapları bul ve öneri listesine ekle
for i, book1 in enumerate(df['title']):
    for j, book2 in enumerate(df['title']):
        if book1 != book2:
            # İlk kitabın ve ikinci kitabın indekslerini bul
            index1 = indices.get(book1, None)
            index2 = indices.get(book2, None)

            # Geçerli indeksler olup olmadığını kontrol et
            if index1 is None or index2 is None or index1 >= cosine_sim.shape[0] or index2 >= cosine_sim.shape[0]:
                continue

            # Benzerlik skorlarını al
            scores1 = cosine_sim[index1]
            scores2 = cosine_sim[index2]

            # İki kitap arasındaki benzerlik skorlarını karşılaştır
            combined_scores = (scores1 + scores2) / 2

            # Kendini ve diğer seçilen kitabı öneri olarak almayalım
            combined_scores[index1] = -1
            combined_scores[index2] = -1

            # En yüksek benzerlik skorlarına sahip 5 kitabı bul
            if len(combined_scores) > 5:
                top_5_indices = np.argpartition(-combined_scores, 5)[:5]
                top_5_indices = top_5_indices[np.argsort(-combined_scores[top_5_indices])]
            else:
                top_5_indices = np.argsort(-combined_scores)[:5]

            # Seçilen kitaplar haricindeki 5 kitabı öneri olarak ekleyelim
            recommendations = [df['title'].iloc[idx] for idx in top_5_indices if idx != index1 and idx != index2]

            # Sadece 5 öneri ile sınırlayalım
            recommendations = recommendations[:5]

            # İki kitap için öneri listesini dictionary'ye ekleyelim
            recommendation_dict[(book1, book2)] = recommendations

# Sözlüğü bir veri çerçevesine dönüştür
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Book Pair', 'Recommendations'])

# Veri çerçevesini kontrol et
print(recommendation_df.head())

recommendation_df.to_excel("data/book_recommendations2.xlsx", index=False)


##########################################################
# Seçilen filme karşılık kitap önerisi
##########################################################

# Veri Setlerini Yükleme
df_book = pd.read_excel("data/books_duzenlendi.xlsx")
df_movie = pd.read_excel("data/movies_duzenlendi.xlsx")

# Veri setimizin Weighted Rating'e göre öne çıkan bölümlerini alıyoruz.
df_book = df_book.head(1000)
df_movie = df_movie.head(1000)

# Kitap ve Film Açıklamalarını Birleştirme
combined_descriptions = pd.concat([df_book['description'], df_movie['Description']], ignore_index=True)

# Ortak TF-IDF Vektörizer Oluşturma
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_descriptions)

# Film ve Kitap TF-IDF Matrislerini Ayırma
tfidf_matrix_movie = tfidf_matrix[len(df_book):]
tfidf_matrix_book = tfidf_matrix[:len(df_book)]

# Cosine Benzerlik Matrisi Oluşturma
cosine_sim = cosine_similarity(tfidf_matrix_movie, tfidf_matrix_book)

# Öneri Listesini Saklayacak Sözlük
recommendation_dict = {}

# Her Film İçin Benzer Kitapları Bul ve Öneri Listesine Ekle
for i, movie in enumerate(df_movie['movie title']):

    # Benzerlik Skorlarını DataFrame'e Dönüştürme
    similarity_scores = pd.DataFrame(cosine_sim[i], columns=['score'])

    # En Yüksek Benzerlik Skorlarına Sahip 5 Kitabı Bulma
    sorted_scores = similarity_scores.sort_values(by='score', ascending=False)
    top_5_indices = sorted_scores.index[:5]
    recommendations = df_book['Book Name'].iloc[top_5_indices].tolist()

    # Öneri Listesine Ekleme
    recommendation_dict[movie] = recommendations

# Sözlüğü Bir Veri Çerçevesine Dönüştürme
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Movie Title', 'Book Recommendations'])

# Veri Çerçevesini Kontrol Etme
print(recommendation_df.head())

# Veri Çerçevesini Excel Dosyasına Kaydetme
recommendation_df.to_excel("data/movie_to_book_recommendations.xlsx", index=False)



##########################################################
# Seçilen kitaba göre film önerisi
##########################################################

df_book = pd.read_excel("data/books_duzenlendi.xlsx")
df_movie = pd.read_excel('data/movies_duzenlendi.xlsx')

# Veri setimizin Weighted Rating'e göre öne çıkan bölümlerini alıyoruz.
df_book = df_book.head(1000)
df_movie = df_movie.head(1000)

df_book.head()
df_movie.head()

# Kitap ve Film Açıklamalarını Birleştirme
# Bu adım, TF-IDF vektörizerin hem kitaplar hem de filmler için aynı kelime dağarcığını öğrenmesini sağlar
combined_descriptions = pd.concat([df_book['description'], df_movie['Description']], ignore_index=True)

# Ortak TF-IDF Vektörizer Oluşturma
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_descriptions)

# Kitap ve Film TF-IDF Matrislerini Ayırma
tfidf_matrix_book = tfidf_matrix[:len(df_book)]
tfidf_matrix_movie = tfidf_matrix[len(df_book):]

# Cosine Benzerlik Matrisi Oluşturma
cosine_sim = cosine_similarity(tfidf_matrix_book, tfidf_matrix_movie)

# Öneri Listesini Saklayacak Sözlük
recommendation_dict = {}

# Her Kitap İçin Benzer Filmleri Bul ve Öneri Listesine Ekle
for i, book in enumerate(df_book['title']):
    # Benzerlik Skorlarını DataFrame'e Dönüştürme
    similarity_scores = pd.DataFrame(cosine_sim[i], columns=['score'])

    # En Yüksek Benzerlik Skorlarına Sahip 5 Filmi Bulma
    sorted_scores = similarity_scores.sort_values(by='score', ascending=False)
    top_5_indices = sorted_scores.index[:5]
    recommendations = df_movie['movie title'].iloc[top_5_indices].tolist()

    # Öneri Listesine Ekleme
    recommendation_dict[book] = recommendations

# Sözlüğü Bir Veri Çerçevesine Dönüştürme
recommendation_df = pd.DataFrame(list(recommendation_dict.items()), columns=['Books', 'Recommended Films'])

# Veri Çerçevesini Kontrol Etme
print(recommendation_df.head())

# Veri Çerçevesini Excel Dosyasına Kaydetme
recommendation_df.to_excel("data/book_to_film_recommendations.xlsx", index=False)