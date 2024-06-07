import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path = 'credit_scores.csv'  # Укажите путь к файлу, если это необходимо
df = pd.read_csv(file_path)

# Удаление указанных признаков
df.drop(columns=["Name", "SSN", "ID", "Customer_ID"], inplace=True)

# Определение целевой переменной и входных переменных
X = df.drop(columns=["Credit_Score"])
y = df["Credit_Score"]

# Разделение набора данных на 80% обучающих и 20% тестовых данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Определение числовых и категориальных столбцов
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Создание конвейера для обработки пропущенных значений и масштабирования числовых данных
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Создание конвейера для обработки пропущенных значений и кодирования категориальных данных
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединение числовых и категориальных конвейеров
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Определение значений для перебора
kernels = ['rbf', 'linear']
C_values = [20, 0.01, 10]

best_accuracy = 0
best_kernel = ''
best_C = 0

# Перебор значений для выбора лучших гиперпараметров
for kernel in kernels:
    for C in C_values:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', SVC(kernel=kernel, C=C))])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Kernel: {kernel}, C: {C}, Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_C = C

print(f"Best Kernel: {best_kernel}, Best C: {best_C}, Best Accuracy: {best_accuracy}")

# Финальное обучение на полном наборе данных с лучшими гиперпараметрами
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SVC(kernel=best_kernel, C=best_C))])

X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
pipeline.fit(X_full, y_full)

# Сохранение модели
joblib.dump(pipeline, 'best_svm_model.pkl')
