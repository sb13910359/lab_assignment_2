const int buttonPin = 2;
bool lastState = HIGH;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {
  bool state = digitalRead(buttonPin);

  if (state == LOW && lastState == HIGH) {
    Serial.println("ESTOP");
    lastState = LOW;
  } 
  else if (state == HIGH && lastState == LOW) {
    lastState = HIGH;
  }

  delay(50);
}
