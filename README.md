# Laporan Proyek Machine Learning - M. Rifal Alfathur Fauzan

## Domain Proyek
proyek ini diangkat dengan tujuan menganalysis apa yang menyebabkan harga mobil di Polandia berbeda
dengan referensi [Tomasz Komornicki](https://www.tandfonline.com/doi/abs/10.1080/01441647.2002.10823175)pada "Factors of development of car ownership in Poland".

## Business Understanding
Anda seorang pemilik showroom mobil di Polandia, perusahaan anada membeli dan menjual mobil bekas. penerapan automasi pada sistem memprediksi harga mobil dengan teknik predictive modeling

semua bisnis mengejar profit. penting bagi perusahaan untuk mengetahui dan dapat memprediksi harga mobil di pasar polandia. prediksi akan digunakan untuk menentukan berapa harga beli yang pantas untuk mobil dengan karakteristik tertentu sehingga perusahaan bisa mendapatkan profit sebesar mungkin.

### Problem Statements
Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga mobil di polandia?
Berapa harga pasar mobil di polandia dengan karakteristik atau fitur tertentu? 

### Goals
Mengetahui fitur yang paling berkorelasi dengan harga mobil di polandia.
Membuat model machine learning yang dapat memprediksi harga mobil di polandia seakurat mungkin berdasarkan fitur-fitur yang ada.

## Data Understanding
Dataset didapat di  [kaggle](https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland). dengan 117.927 dengan berbagai karakteristik dan harga. Dengan karakteristik non-numerik mark, model, generation_name, city, province. dan karakteristik numerik year, mileage, vol_engine, city. Fitur-fitur ini yang akan digunakan dalam menemukan pola pada data, sedangkan harga fitur target.


Berdasarkan informasi dari Kaggle, variabel-variabel yang digunakan pada dataset adalah sebagai berikut:
mark = car mark
model = car model
generation_name = Formatted Generation
year = Car Year
mileage = Car Mileage in Kilometers
vol_engine = Auto Engine Size
fuel = Engine Type
city = locality in Poland
province = Region of Poland
price = Price in PLN (approx. 1USD=1PLN)

## Exploratory Data Analysis
### Deskripsi Variabel
ketika cek informasi data terdapat:
- terdapat empat kolom int64: year, mileage, vol_engine, dan price
- terdapat enam kolom object: mark, model, generation_name, fuel, city, dan province

dan ketika dicek desribisi data terdapat:
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas
- interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.


### Menangani Missing Value
terdapat missing value di vol_engine karena volume engine tidak mungkin 0 tidak akan 0 sehingga dapat dihilangkan, dan sama untuk mileage karena bahkan untuk setiap mobil baru pun tidak akan 0.

Selanjutnya kita tangani masalah missing value ini dengan menggantinya dengan median karena jika dihapus itu tidak mungkin karena data yang harus dihapus cukup banyak dan itu dapat mempengaruhi dataset.

### Menangani Outliers
persamaaan untuk mencari batas bawah dan atas adalah sebagai berikut:

- Batas bawah = Q1 - 1.5 * IQR
- Batas atas = Q3 + 1.5 * IQR
data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1

### Univariate Analysis
Analysis dengan teknik univariate EDA. pertama tama :membagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features. Dan untuk analysis memfokuskan kepada variable "price" 

### Exploratory Data Analysis - Multivariate Analysis
Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data.

#### Categorical Features
Pada tahap ini, kita akan mengecek rata-rata harga terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap harga

#### Numerical Features
mengobservasi korelasi antara fitur numerik dengan fitur target

## Data Preparation
### Encoding Fitur Kategori
teknik yang digunakan untuk encoding adalah one-hot-encoding. Penggunaan Library scikit-learn untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori antara lain, 'mark', 'model', 'generation_name', 'fuel', 'city', 'province' proses encoding ini menggunakanfitur get_dummmies

### Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) hal ini dilakukan sebelum membuat model. Porses ini dilakukan sebelum proses scaling (penyekalaan) bertujuan agar tidak terjadinya kebiciran data (data leakage). Maka proses scaling dilakukan terpisah antara data latih dan data uji.

Pada proyek ini memiliki proporsi pembagian data latih dan data uji sebanyak 90:10 hal ini dikarena dataset yang digunakan berjumlah banyak, dengan random_state = 42.

### Standarisasi
Proses standardisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini menggunakan teknik StandarScaler dari library scikitlearn.

StandarScaler melakukan proses standardisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Penerapan fitur dilakukan pada data latih saja dan untuk data uji akan di lakukan pada tahap evaluasi.

proses standarisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.

## Model Development
Pada proyek ini menggunakan algoritma machine learning KNN, Random Forest, dan Boosting. mean_squared_error sebagai metrik untuk mengevaluasi performa model

### Random Forest
random forest pada dasarnya adalah versi bagging dari algoritma decision tree. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement.

- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

### Boosting
algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. Dilihat dari caranya memperbaiki kesalahan pada model sebelumnya, algoritma boosting terdiri dari dua metode:
- Adaptive boosting
- Gradient boosting
Proyek ini menggunakan metode adaptive boosting. Salah satu metode adaptive boosting yang terkenal adalah AdaBoost

Berikut merupakan parameter-parameter yang digunakan pada potongan kode di atas.

learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
random_state: digunakan untuk mengontrol random number generator yang digunakan.

### KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus regresi.

Kelebihan KNN adalah algoritma yang mudah digunakan dan dipahami, namun memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar.

Pada proyek ini nilai ketetanggaan terdekat(K) adalah 10 dan metode penghitungan jarang nya menggunakan default yaitu minkowski.

## Evaluasi
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. tapi terlebih dahulu data uji discaling, hal ini dilakukan agar skala antara data latih dan data uji sama.
