import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle

class ChurnModel:
    def __init__(self, data, target, features):
        self.data = data
        self.target = target
        self.features = features
        self.x = self.data[self.features]
        self.y = self.data[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.model = None
        self.prediction = None
        
    def handle_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        self.data.loc[self.data[column] < lower_bound, column] = lower_bound
        self.data.loc[self.data[column] > upper_bound, column] = upper_bound
        
    def preprocess_data(self):
        outlier_columns = ['CreditScore', 'Age', 'NumOfProducts', 'HasCrCard']
        for column in outlier_columns:
            self.handle_outliers(column)
    
    def train_model(self):
        self.model = XGBClassifier(random_state=42)
        self.model.fit(self.x_train, self.y_train)
        
    def evaluate_model(self):
        self.prediction = self.model.predict(self.x_test)
        print("Classification Report:\n", classification_report(self.y_test, self.prediction))
        
    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)
            
    def load_model(self, filename):
        with open(filename, 'rb') as model_file:
            self.model = pickle.load(model_file)
    
    def predict(self, new_data):
        return self.model.predict(new_data)

if __name__ == "__main__":
    #Memuat dataset
    df = pd.read_csv(r"C:\Users\Fiona\Documents\coolyeah\Semester 4\Model Deployment\UTS_kumpul\data_D.csv")

    #Memilih fitur-fitur yang akan digunakan
    selected_features = ['Age', 'NumOfProducts', 'IsActiveMember']    
    target = 'churn'

    #Membuat objek model
    model = ChurnModel(data=df, target=target, features=selected_features)

    #Melakukan preprocessing data
    model.preprocess_data()

    #Melatih model
    model.train_model()

    #Evaluasi model
    model.evaluate_model()

    #Menyimpan model terbaik
    model.save_model('best_model.pkl')

    #Memuat kembali model terbaik
    loaded_model = ChurnModel(data=df, target=target, features=selected_features)
    loaded_model.load_model('best_model.pkl')

    #Contoh data untuk prediksi
    new_data = pd.DataFrame({
        'Age': [57],
        'NumOfProducts': [2],
        'IsActiveMember': [1]
    })

    #Melakukan prediksi dengan model yang telah dimuat
    prediction = loaded_model.predict(new_data)

    #Menampilkan hasil prediksi
    print("Hasil prediksi:", prediction)

