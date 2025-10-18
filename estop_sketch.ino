const int buttonPin = 2;      // Connect button to digital pin 2
bool lastState = HIGH;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);      // Internal pull up resistor
  Serial.begin(9600);
}

void loop() {
  bool state = digitalRead(buttonPin);

  if (state == LOW && lastState == HIGH) {
    Serial.println("ESTOP");          // prints "ESTOP" for system to read via serial
    lastState = LOW;
  } 
  else if (state == HIGH && lastState == LOW) {
    lastState = HIGH;                  // Makes sure ESTOP is only triggered once for one button press
  }

  delay(50);
}
