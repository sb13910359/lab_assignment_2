const int buttonPins[3] = {2, 3, 4};   // Button1: Digital pin2, Button2:digital pin3, Button3:Digital pin4
bool lastStates[3] = {HIGH, HIGH, HIGH};  
int buttonStage[3] = {0, 0, 0};           // 0=normal, 1= estopactive, 2=confirm release estop 

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
      buttonStage[i] = (buttonStage[i] + 1) % 3;    //change the button stage

      if (buttonStage[i] == 1) {
        Serial.print("ESTOP_R");   // prints "ESTOP" for system to read via serial
        Serial.println(i + 1);     // e.g ESTOP_R1 / ESTOP_R2 / ESTOP_R3
      } 

      else if (buttonStage[i] == 2) {
        Serial.print("CONFIRM_R");  // confirm release?
        Serial.println(i + 1);    
      }

      else if (buttonStage[i] == 0) {
        Serial.print("CLEAR_R");  // Clear estop
        Serial.println(i + 1);   
      }

      lastStates[i] = LOW;
    }

    else if (state == HIGH && lastStates[i] == LOW) {
      lastStates[i] = HIGH;                          // Makes sure ESTOP is only triggered once for one button press
    }
  }

  delay(50);
}
