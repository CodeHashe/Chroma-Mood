# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,Input
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# import joblib

# train=pd.read_table('train.txt', delimiter = ';', header=None, )
# val=pd.read_table('val.txt', delimiter = ';', header=None, )
# test=pd.read_table('test.txt', delimiter = ';', header=None, )

# df = pd.concat([train ,  val , test])
# df.columns = ["content", "sentiment"]
# LE=LabelEncoder()
# df['N_label'] = LE.fit_transform(df['sentiment'])
# cv = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
# df_cv = cv.fit_transform(df['content']).toarray()

# X_train, X_test, y_train, y_test =train_test_split(df_cv, df['N_label'], test_size=0.25, random_state=42,stratify=df['N_label'])
# num_classes = len(np.unique(y_train))
# # print(num_classes)
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))
# model = Sequential()

# model.add(Input(shape=(X_train.shape[1],)))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))

# # Second hidden block
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))

# # Third hidden block
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))

# # Fourth hidden block
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))

# # Output layer
# model.add(Dense(num_classes, activation='softmax'))

# # compile the keras model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # fit the keras model on the dataset
# model.fit(X_train, y_train,validation_split=0.1, epochs=20, batch_size=32,class_weight=class_weights,callbacks=[early_stop])


# train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}")

# #save the model
# joblib.dump(cv, "vectorizer.pkl")
# joblib.dump(LE, "label_encoder.pkl")
# model.save("TexttoEmotion.h5")



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib
import mlflow
import mlflow.keras

train = pd.read_table('train.txt', delimiter=';', header=None)
val = pd.read_table('val.txt', delimiter=';', header=None)
test = pd.read_table('test.txt', delimiter=';', header=None)

df = pd.concat([train, val, test])
df.columns = ["content", "sentiment"]


LE = LabelEncoder()
df['N_label'] = LE.fit_transform(df['sentiment'])

cv = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
df_cv = cv.fit_transform(df['content']).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    df_cv, df['N_label'], test_size=0.25, random_state=42, stratify=df['N_label']
)

num_classes = len(np.unique(y_train))
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

mlflow.set_experiment("Text_Emotion_Classification")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("ngram_range", "1-2")
    mlflow.log_param("stop_words", "english")
    mlflow.log_param("dense_layers", [512, 256, 128, 64])
    mlflow.log_param("dropout_rate", 0.4)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 20)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop]
    )


    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}")

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("test_loss", test_loss)

    
    joblib.dump(cv, "vectorizer.pkl")
    joblib.dump(LE, "label_encoder.pkl")
    model.save("TexttoEmotion.h5")

    mlflow.log_artifact("vectorizer.pkl")
    mlflow.log_artifact("label_encoder.pkl")
    mlflow.keras.log_model(model, "TexttoEmotion_model")