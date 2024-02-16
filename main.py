from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x:  '%.3f' % x)


# fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

# metadata
print(breast_cancer_wisconsin_original.metadata)

# variable information
print(breast_cancer_wisconsin_original.variables)

# dataframe oluşturma
y_df = pd.DataFrame(y, columns=["Class"])
df = pd.concat([X, y_df], axis=1)

# EDA
def check_df(dataframe):
    print('#############SHAPE#################')
    print(dataframe.shape)
    print('#############TYPES#################')
    print(dataframe.dtypes)
    print('################HEAD#################')
    print(dataframe.head())
    print('###############TAIL###################')
    print(dataframe.tail())
    print('###############NA######################')
    print(dataframe.isnull().sum())
    print('################QUANTILES################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#############################################################')
    if plt:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)


    def num_summary(dataframe, num_cols, plot=True):
        for col in num_cols:
            print("{} için temel istatistikler:".format(col))

            quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
            print(dataframe[col].describe(quantiles).T)

            if plot:
                dataframe[col].hist(bins=20)
                plt.xlabel(col)
                plt.title(col)
                plt.show(block=True)

num_summary(df, num_cols)

# ## Numeric değişkenlerin target göre analizi
def target_summary_with_num(dataframe, target, col):
    print(dataframe.groupby(target).agg({col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Class", col)

# KATEGORİK DEĞİŞKENLERİN TARGRET GÖRE ANALİZİ
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "COUNT": dataframe[categorical_col].value_counts(),
                        "RATIO": 100* dataframe[categorical_col].value_counts() / len(dataframe)}),end="\n\n\n")

for col in cat_cols:
    if col != "Class":
        target_summary_with_cat(df, "Class", col)
# Mitoses arttıkça kötü huyluluk da artmış fakat oranında gitgide azaldığını görüyoruz

df[num_cols].corr()
#Korelasyon matrisi
f, ax =plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Class ı da num_cols a ekleyip corelasyonuna bakalım
num_cols.append('Class')
df[num_cols].corr()
#Korelasyon matrisi
f, ax =plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

df.corrwith(df['Class']).sort_values(ascending=False)
# Class ile aralarındaki ilişkileri güçlü olanlar:
#Bare_nuclei              -->     0.823
#Uniformity_of_cell_shape -->     0.819
#Uniformity_of_cell_size  -->     0.818

# FEATURE ENGİNEERİNG
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)
#             n_miss  ratio
#Bare_nuclei      16  2.290

from scipy.stats import chi2_contingency
def missing_values_analysis(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    for col in na_columns:
        # Eksik değer olup olmamasına göre bir sütun oluştur
        dataframe[col + '_is_missing'] = np.where(dataframe[col].isnull(), 1, 0)

    # Eksik değer içeren her bir sütun için
    for col in na_columns:
        print(f"\n{col} için eksiklik analizi:")
        for other_col in dataframe.columns.difference(na_columns):
            table = pd.crosstab(dataframe[other_col], dataframe[col + '_is_missing'])
            chi2, p, dof, expected = chi2_contingency(table)
            if p < 0.05:
                print(f"{other_col} sütunu ile {col} arasında istatistiksel olarak anlamlı bir ilişki var. (p={p:.4f})")
            else:
                print(f"{other_col} sütunu ile {col} arasında anlamlı bir ilişki yok. (p={p:.4f})")
        dataframe.drop(columns=[col + '_is_missing'], inplace=True)

missing_values_analysis(df)
# test sonucuna göre eksik değerin veri tipi MCAR'dır.
# eksik değerleri median ile dolduracağız aykırılıktan daha az etkilenmek için

corr_value = df['Bare_nuclei'].corr(df['Class'])
print(f'"Bare_nuclei" ve "Class" arasındaki korelasyon: {corr_value}')
# kanserin iyi mi kötü mü huylu olduğunu belirlemede önemli etkisi var Bare__nuclei'nin
# Eksik değerler imputasyon yöntemi ile doldurulacaktır
# Model tabanlı imputasyon yöntemleri, diğer değişkenlerin bilgilerini kullanarak
# eksik değerleri daha doğru bir şekilde tahmin etmeye çalışır ve
# bu özelliğin hedef değişkenle olan ilişkisini korumaya yardımcı olabilir.

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
dff = df.copy()
# Model tabanlı imputasyon için IterativeImputer kullanımı
imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
# Eksik değerlerin doldurulması
dff["Bare_nuclei"] = imputer.fit_transform(df[["Bare_nuclei"]])

# İmputasyon sonrası kontrol
missing_values_table(dff)
dff["Bare_nuclei"].corr(df["Class"]) #  0.8160499264435999
# korelasyon 0.81 e düştü medyanla doldurma yapılacak.
# medyan ile doldurma
df["Bare_nuclei"].fillna(df["Bare_nuclei"].median(), inplace=True)
missing_values_table(df)
df["Bare_nuclei"].corr(df["Class"]) #  0.8189679369613876
# medyanla doldurduğumuzda korelasyon az bir farkla daha yüksek çıktı imputasyona göre
# ve ayrıca aykırı değerlerden de az etkilendiğinden dolayı medyan ile doldurmayı tercih ettim
del dff

# Aykırı değer kontrolü
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# TEK DEĞİŞKENLİ İÇİN AYKIRI DEĞER ANALİZİ
for col in num_cols:
    if check_outlier(df, col) is True:
        print(f'{col} sütununda aykırı değer vardır.')
# Marginal_adhesion sütununda aykırı değer vardır.
# Single_epithelial_cell_size sütununda aykırı değer vardır.
# Bland_chromatin sütununda aykırı değer vardır.
# Normal_nucleoli sütununda aykırı değer vardır.

for col in cat_cols:
    if check_outlier(df, col) is True:
        print(f'{col} sütununda aykırı değer vardır.')
# Mitoses numerik fakat kategorik değişkenin  sütununda aykırı değer vardır.


outliers_numerical = ['Marginal_adhesion',' Single_epithelial_cell_size','Bland_chromatin','Normal_nucleoli']
outliers_categorical = ['Mitoses']
# Görselleştirme
def plot_box_and_scatter(dataframe, col_name):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))  # 1 satır ve 2 sütunluk subplot oluşturur.

    # Box Plot
    sns.boxplot(data=dataframe, y=col_name, ax=ax[0])
    ax[0].set_title(f'Box Plot for {col_name}')
    ax[0].set_ylabel(col_name)

    # Scatter Plot
    sns.scatterplot(data=dataframe, x=dataframe.index, y=col_name, ax=ax[1])
    ax[1].set_title(f'Scatter Plot for {col_name}')
    ax[1].set_ylabel(col_name)
    ax[1].set_xlabel('Index')

    plt.tight_layout()  # Plotlar arası uygun boşluk bırakır.
    plt.show()

all_outliers = outliers_numerical + outliers_categorical

# Veri setiniz 'df' adıyla tanımlı olmalıdır. Eğer farklı bir isimdeyse, 'df' yerine o ismi kullanın.
for col_name in all_outliers:
    col_name = col_name.strip() # Başında veya sonunda boşluk olabilir, temizleyelim
    plot_box_and_scatter(df, col_name)
# scatter plot grafiklerinde verilerin dağılımının geniş olduğunu gördük.
def outlier_info(dataframe, all_outliers):
    # Aykırı değer sayısı ve oranını hesaplayacak fonksiyonları içerecek.
    def count_outliers(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
        return outliers.shape[0]

    outlier_info = []

    for col_name in all_outliers:
        col_name = col_name.strip()  # Başında veya sonunda boşluk olabilir, temizleyelim
        num_outliers = count_outliers(dataframe, col_name)
        total_count = dataframe[col_name].count()
        outlier_ratio = num_outliers / total_count
        outlier_info.append((col_name, num_outliers, total_count, outlier_ratio))

    # Sütun ismi, aykırı değer sayısı, toplam sayı ve oranı içeren bir DataFrame oluştur.
    outlier_info_df = pd.DataFrame(outlier_info, columns=['Column', 'OutlierCount', 'TotalCount', 'OutlierRatio'])

    # Orana göre azalan sırayla sırala.
    sorted_outlier_info_df = outlier_info_df.sort_values(by='OutlierRatio', ascending=False).reset_index(drop=True)

    return sorted_outlier_info_df

outlier_info(df, all_outliers)

#                         Column  OutlierCount  TotalCount  OutlierRatio
# 0                      Mitoses           120         699         0.172
# 1              Normal_nucleoli            77         699         0.110
# 2            Marginal_adhesion            60         699         0.086
# 3  Single_epithelial_cell_size            54         699         0.077
# 4              Bland_chromatin            20         699         0.029


#İKİLİ AYKIRI DEĞER ANALİZİ
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from scipy.stats import f_oneway


def perform_bivariate_analysis(dataframe, numerical_cols, categorical_cols, target_variable):
    # Sayısal Değişkenler için Korelasyon Analizi
    print("Korelasyon Analizi:")
    correlation_matrix = dataframe[numerical_cols].corr()
    print(correlation_matrix)

    # Sayısal Değişkenler için Scatter Plotlar
    print("\nSayısal Değişkenler için Scatter Plotlar:")
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            sns.scatterplot(data=dataframe, x=numerical_cols[i], y=numerical_cols[j])
            plt.title(f'{numerical_cols[i]} vs {numerical_cols[j]}')
            plt.show()

    # Sayısal ve Kategorik Değişkenler için Box Plotlar
    print("\nKategorik ve Sayısal Değişkenler Arası İlişkiler için Box Plotlar:")
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            sns.boxplot(x=dataframe[cat_col], y=dataframe[num_col])
            plt.title(f'Box plot of {num_col} by {cat_col}')
            plt.show()

    # Kategorik Değişkenler için Çapraz Tablolar ve Chi-Kare Testi
    print("\nKategorik Değişkenler için Çapraz Tablolar ve Chi-Kare Testi:")
    for cat_col in categorical_cols:
        contingency_table = pd.crosstab(dataframe[cat_col], dataframe[target_variable])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Çapraz Tablo ve Chi-Kare Testi için {cat_col}:")
        print(f"Chi-Kare Değeri: {chi2}, p-Değeri: {p}\n")

    # Sayısal Değişkenlerin Kategorik Değişkenlere Göre İstatistiksel Testler
    print("\nSayısal Değişkenlerin Kategorik Değişkenlere Göre İstatistiksel Testler:")
    for num_col in numerical_cols:
        groups = dataframe.groupby(target_variable)[num_col].apply(list)
        f, p = f_oneway(*groups)
        print(f"{num_col} için ANOVA testi: F-istatistiği = {f}, p-değeri = {p}")

perform_bivariate_analysis(df, num_cols, cat_cols, "Class")

# Aykırı değerleri eşik değerleri ile doldurma
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    if check_outlier(df,col) == True:
        replace_with_thresholds(df, col)


for col in df.columns:
    if check_outlier(df, col) == True:
        print("Aykırı değer var")
    else:
        print("Aykırı değer yok")
# Aykırı değer kontrolü yapıldı ,aykırı değerler yok

#Feature Extraction
# oluşturduğum özelliklerin başına new ekledim diğer orijinal özelliklerle karışmaması için
df['new_complex_feature'] = (df['Clump_thickness'] * df['Marginal_adhesion']) / (df['Bland_chromatin'] + 1)
df['new_interaction_size_adhesion'] = df['Uniformity_of_cell_size'] * df['Marginal_adhesion']

df['new_Cell_Size_Shape_Avg_Uniformity'] = (df['Uniformity_of_cell_size'] + df['Uniformity_of_cell_size']) / 2
df['new_Total_Uniformity_Score'] = df['Uniformity_of_cell_size'] + df['Uniformity_of_cell_size'] + df['Marginal_adhesion']
df['new_Epithelial_BareNuclei_Interaction'] = df['Single_epithelial_cell_size'] * df['Bare_nuclei']

########################################################################################################################
######################################### MODELLEME ####################################################################
#######################################################################################################################
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

##########################################       XGBOOST CLASSIFICATION                #################################
# Hedef değişkenin dönüştürülmesi: 2 -> 0 (benign), 4 -> 1 (malignant)
df['Class'] = df['Class'].replace({2: 0, 4: 1})

X = df.drop('Class', axis=1)
y = df['Class']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri dengesizliğinin giderilmesi için SMOTE uygulama
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# SMOTE sonrası sınıf dengesini kontrol etme
print("Sınıf dengesi:", np.bincount(y_train_res))

# XGBoost sınıflandırıcı modelinin oluşturulması
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Aranacak hiperparametrelerin belirlenmesi
param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5],
    'scale_pos_weight': [1, 2, 3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Recall skorunu optimize etmek için scorer oluştur
recall_scorer = make_scorer(recall_score)

# GridSearchCV nesnesinin oluşturulması
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=recall_scorer, n_jobs=-1, cv=StratifiedKFold(n_splits=5), verbose=3)

# GridSearchCV ile modelin eğitilmesi
grid_search.fit(X_train_res, y_train_res)

# En iyi hiperparametrelerin ve en iyi recall skorunun gösterilmesi
print("En iyi hiperparametreler:", grid_search.best_params_)
print("En iyi recall skoru:", grid_search.best_score_)

# En iyi model ile test seti üzerinde tahmin yapma
xgboost_best_model = grid_search.best_estimator_
y_pred = xgboost_best_model.predict(X_test)

# Modelin performansını değerlendirme
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"ROC AUC Score: {roc_auc:.6f}")

evaluate_model(y_test, y_pred)
# En iyi recall skoru: 0.994482496194825
# Accuracy: 0.985714
# Precision: 0.957447
# Recall: 1.000000
# F1 Score: 0.978261
# ROC AUC Score: 0.989474

def plot_importance_for_xboost(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance_for_xboost(xgboost_best_model, X_train)

###############################         LOGISTIC REGRESSION        #####################################################
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression modelini oluşturma ve eğitme
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_res_scaled, y_train_res)

# Test seti üzerinde tahmin yapma
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# Modelin performansını değerlendirme
evaluate_model(y_test, y_pred_log_reg)
# Accuracy: 0.971429
# Precision: 0.976744
# Recall: 0.933333
# F1 Score: 0.954545
# ROC AUC Score: 0.961404

def plot_importance_for_logistic_regression(model, features):
    """
    Logistic Regression modelinin katsayılarını görselleştirir.

    Parametreler:
    - model: Eğitilmiş Logistic Regression modeli.
    - features: Özelliklerin isimlerini içeren liste veya dizi.
    """
    # Model katsayılarını DataFrame'e dönüştür
    coef = pd.DataFrame(model.coef_.flatten(), index=features.columns, columns=['Coefficients'])

    # Katsayıların mutlak değerlerine göre sırala
    coef['abs'] = coef['Coefficients'].abs()
    sorted_coef = coef.sort_values(by='abs', ascending=False).drop('abs', axis=1)

    # Görselleştirme
    plt.figure(figsize=(10, len(features.columns) / 2))
    sns.barplot(x="Coefficients", y=sorted_coef.index, data=sorted_coef)
    plt.title('Logistic Regression Feature Importances')
    plt.tight_layout()
    plt.show()

plot_importance_for_logistic_regression(log_reg_model, pd.DataFrame(X_train_res_scaled, columns=X_train.columns))

################################        Random Forest Classification     ###############################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint as sp_randint
import numpy as np

# Random Forest modelinin oluşturulması
rf_model = RandomForestClassifier(random_state=42)

# Modelin eğitim verileriyle eğitilmesi
rf_model.fit(X_train_res, y_train_res)

# Test veri seti üzerinde tahminlerin yapılması
y_pred_rf = rf_model.predict(X_test)

# Model performansının değerlendirilmesi
evaluate_model(y_test, y_pred_rf)

# Hiperparametrelerin rastgele seçim için aralıklarının tanımlanması
param_dist = {
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "bootstrap": [True, False],
    "n_estimators": sp_randint(100, 500)
}

# RandomizedSearchCV nesnesinin oluşturulması
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, scoring='recall')

# Modelin RandomizedSearchCV ile eğitilmesi
random_search.fit(X_train_res, y_train_res)

# En iyi parametrelerin ve skorun gösterilmesi
print("En iyi parametreler:", random_search.best_params_)
print("En iyi skor:", random_search.best_score_)

# En iyi model ile tahmin yapma ve performansının değerlendirilmesi
y_pred_best_random = random_search.best_estimator_.predict(X_test)
evaluate_model(y_test, y_pred_best_random)
# Accuracy: 0.978571
# Precision: 0.956522
# Recall: 0.977778
# F1 Score: 0.967033
# ROC AUC Score: 0.978363

def plot_feature_importances(model, features):
    importances = pd.DataFrame({'feature': features.columns, 'importance': model.feature_importances_})
    importances.sort_values('importance', ascending=False, inplace=True)

    # Özellik önemlerini çizdir
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importances')
    plt.tight_layout()

plot_feature_importances(rf_model, X_train)

################################# SUPPORT VECTOR MACHINE ###############################################################
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri ön işleme: Özellik ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC modelinin oluşturulması
svc_model = SVC(kernel='linear', random_state=42)

# Modelin eğitilmesi
svc_model.fit(X_train_scaled, y_train)

# Test veri seti üzerinde tahminlerin yapılması
y_pred = svc_model.predict(X_test_scaled)

# Model performansının değerlendirilmesi
print("SVC Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hiperparametre ayarlama: GridSearchCV kullanımı
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)

# GridSearchCV ile bulunan en iyi modelle test veri seti üzerinde tahmin yapma
y_pred_best = grid_search.best_estimator_.predict(X_test_scaled)
print("GridSearchCV ile SVC Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
#               precision    recall  f1-score   support
#            0       0.97      0.98      0.97        95
#            1       0.95      0.93      0.94        45
#     accuracy                           0.96       140
#    macro avg       0.96      0.96      0.96       140
# weighted avg       0.96      0.96      0.96       140
evaluate_model(y_test, y_pred_best)
# Accuracy: 0.964286
# Precision: 0.954545
# Recall: 0.933333
# F1 Score: 0.943820
# ROC AUC Score: 0.956140

##########################################      NEURAL NETWORK      ####################################################
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Örnek veri seti yüklenir
# X, y = load_your_data()

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri ön işleme: Özellik ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE ile veri setinin dengelenmesi
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# Yapay sinir ağı modelinin oluşturulması
model = Sequential()
model.add(Dense(64, input_dim=X_train_res.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Modelin derlenmesi
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(X_train_res, y_train_res, epochs=100, batch_size=10, validation_split=0.2, verbose=2)

# Test veri seti üzerinde tahminlerin yapılması
y_pred = model.predict(X_test_scaled)
y_pred = np.round(y_pred).astype(int)

# Model performansının değerlendirilmesi
print("Neural Network Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Neural Network Model Accuracy: 0.9619047619047619
#               precision    recall  f1-score   support
#            0       0.97      0.98      0.97       143
#            1       0.95      0.93      0.94        67
#     accuracy                           0.96       210
#    macro avg       0.96      0.95      0.96       210
# weighted avg       0.96      0.96      0.96       210
evaluate_model(y_test, y_pred)
# Accuracy: 0.961905
# Precision: 0.953846
# Recall: 0.925373
# F1 Score: 0.939394
# ROC AUC Score: 0.952197
# ROC AUC Score: 0.945029

# Creating a DataFrame with the model performances
model_performance = pd.DataFrame({
    "Model": ["XGBoost", "Logistic Regression", "Random Forest", "SVM", "Neural Network"],
    "Accuracy": [0.985714, 0.971429, 0.978571, 0.964286, 0.961905],
    "Precision": [0.957447, 0.976744, 0.956522, 0.954545, 0.953846],
    "Recall": [1.000000, 0.933333, 0.977778, 0.933333, 0.925373],
    "F1 Score": [0.978261, 0.954545, 0.967033, 0.943820, 0.939394],
    "ROC AUC Score": [0.989474, 0.961404, 0.978363, 0.956140, 0.952197]
})

model_performance

#                  Model  Accuracy  Precision  Recall  F1 Score  ROC AUC Score
# 0              XGBoost     0.986      0.957   1.000     0.978          0.989
# 1  Logistic Regression     0.971      0.977   0.933     0.955          0.961
# 2        Random Forest     0.979      0.957   0.978     0.967          0.978
# 3                  SVM     0.964      0.955   0.933     0.944          0.956
# 4       Neural Network     0.962      0.954   0.925     0.939          0.952