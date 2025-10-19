const int buttonPins[3] = {2, 3, 4};   // Button1: Digital pin2, Button2:digital pin3, Button3:Digital pin4

bool lastStates[3] = {HIGH, HIGH, HIGH};

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 3; i++) {
    pinMode(buttonPins[i], INPUT_PULLUP);   // Internal pull up resistor
  }
}

void loop() {
  for (int i = 0; i < 3; i++) {
    bool state = digitalRead(buttonPins[i]);

    if (state == LOW && lastStates[i] == HIGH) {
      Serial.print("ESTOP_R");    // prints "ESTOP" for system to read via serial
      Serial.println(i + 1);      // e.g. ESTOP_R1 / ESTOP_R2 / ESTOP_R3
      lastStates[i] = LOW;
    } 
    else if (state == HIGH && lastStates[i] == LOW) {
      lastStates[i] = HIGH;      // Makes sure ESTOP is only triggered once for one button press
    }
  }

  delay(50);
}
