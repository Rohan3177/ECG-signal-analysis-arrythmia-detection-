Automatic ECG Monitoring and Classification System
Table of Contents
Introduction
ESP8266 Code
Python Code
Setup and Installation
Usage
License
Acknowledgements
Contributing
Contact
<a name="introduction"></a>

Introduction
This project consists of an ECG monitoring system that uses an ESP8266 microcontroller to send data to Ubidots and a Python script to classify ECG signals using an Artificial Neural Network (ANN).

<a name="esp8266-code"></a>

ESP8266 Code
This code sets up an ESP8266 to connect to WiFi and send ECG data to Ubidots using MQTT.

cpp
Copy code
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
  
#define WIFISSID "Avinash" // Put your WifiSSID here
#define PASSWORD "Avinash@12345" // Put your wifi password here
#define TOKEN "BBUS-19311qlVLH8g245EjthL2fGMjFuQye" // Put your Ubidots' TOKEN
#define MQTT_CLIENT_NAME "myecgsensor" // MQTT client Name
  
/****************************************
 * Define Constants
 ****************************************/
#define VARIABLE_LABEL "myecg" // Assign the variable label
#define DEVICE_LABEL "esp8266" // Assign the device label
#define SENSOR A0 // Set the A0 as SENSOR
  
char mqttBroker[] = "industrial.api.ubidots.com";
char payload[100];
char topic[150];
char str_sensor[10];
  
WiFiClient ubidots;
PubSubClient client(ubidots);
  
void callback(char* topic, byte* payload, unsigned int length) {
  char p[length + 1];
  memcpy(p, payload, length);
  p[length] = NULL;
  Serial.write(payload, length);
  Serial.println(topic);
}
  
void reconnect() {
  while (!client.connected()) {
    Serial.println("Attempting MQTT connection...");
    if (client.connect(MQTT_CLIENT_NAME, TOKEN, "")) {
      Serial.println("Connected");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 2 seconds");
      delay(2000);
    }
  }
}
  
void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFISSID, PASSWORD);
  pinMode(SENSOR, INPUT);
  Serial.println();
  Serial.print("Waiting for WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("");
  Serial.println("WiFi Connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  client.setServer(mqttBroker, 1883);
  client.setCallback(callback);
}
  
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  sprintf(topic, "%s%s", "/v1.6/devices/", DEVICE_LABEL);
  sprintf(payload, "%s", "");
  sprintf(payload, "{\"%s\":", VARIABLE_LABEL);
  float myecg = analogRead(SENSOR);
  dtostrf(myecg, 4, 2, str_sensor);
  sprintf(payload, "%s {\"value\": %s}}", payload, str_sensor);
  Serial.println("Publishing data to Ubidots Cloud");
  client.publish(topic, payload);
  client.loop();
  delay(10);
}
<a name="python-code"></a>

Python Code
This Python script loads ECG data, trains an ANN model to classify ECG signals, and predicts arrhythmias.

python
Copy code
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the ECG dataset
data = pd.read_csv("C:\\ECG_dataset\\archive (1)\\108.csv")

# Features (X): "time_ms," "MLII," and "V5"
X = data[["time_ms", "MLII", "V1"]]

# Create a binary target variable (Y) based on a condition in "MLII"
threshold_value = -0.14
data['target_variable'] = (data['MLII'] < threshold_value).astype(int)

# Target variable (Y)
y = data['target_variable']

# Normalize the features
X = X.astype("float32")
X /= np.max(X, axis=0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu', kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save("ECG_classifier.h5")

# Load the saved model
loaded_model = load_model("ECG_classifier.h5")

# Predict the arrhythmia disease for new data
new_data = pd.read_csv("C:\\ECG_dataset\\archive (1)\\109.csv")

# Check column names
print(new_data.columns)

new_X = new_data[["time_ms", "MLII", "V1"]]
new_X = new_X.astype("float32")
new_X /= np.max(new_X, axis=0)

new_predictions = loaded_model.predict(new_X)

# Convert predictions to binary values based on a threshold for new data
threshold = 0.5
new_binary_predictions = [1 if pred > threshold else 0 for pred in new_predictions]

# Print the predictions for new data
print("Predictions for new data:")
for prediction in new_binary_predictions:
    if prediction == 1:
        print("Arrhythmia detected")
    else:
        print("No arrhythmia detected")

# Calculate precision, recall, and F1-score on the validation set
val_predictions = loaded_model.predict(X_val)
val_binary_predictions = [1 if pred > threshold else 0 for pred in val_predictions]

precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_binary_predictions, average='micro', zero_division=1)

print(f"\nPrecision on validation set: {precision}")
print(f"Recall on validation set: {recall}")
print(f"F1-score on validation set: {f1}")

# Calculate confusion matrix on the validation set
conf_matrix = confusion_matrix(y_val, val_binary_predictions)

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Check unique values in model predictions for validation set
print("Unique values in val_binary_predictions:", np.unique(val_binary_predictions))

# Check the shape of training and validation sets
print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)

# Check class distribution in the target variable for validation set
print("\nClass distribution in the target variable (validation set):")
print(y_val.value_counts())
<a name="setup-and-installation"></a>

Setup and Installation
ESP8266
Install the Arduino IDE from here.
Install the ESP8266 board in the Arduino IDE by going to File > Preferences, and add the following URL in the "Additional Boards Manager URLs" field: http://arduino.esp8266.com/stable/package_esp8266com_index.json.
Go to Tools > Board > Boards Manager, search for ESP8266, and install it.
Connect your ESP8266 to your computer and upload the provided code.
Python
Install Python from here.
Install the necessary libraries:
bash
Copy code
pip install numpy pandas tensorflow scikit-learn
Run the provided Python script.
<a name="usage"></a>

Usage
Set up the hardware components as per the circuit diagram.
Upload the ESP8266 code to the microcontroller using the Arduino IDE.
Ensure the ESP8266 is connected to the internet.
Run the Python script to classify ECG signals.
