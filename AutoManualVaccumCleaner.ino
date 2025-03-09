#include <Servo.h>          // Servo motor library. This is standard library
#include <NewPing.h>        // Ultrasonic sensor function library. You must install this library

// L298N Motor Driver control pins
const int LeftMotorForward = 5;
const int LeftMotorBackward = 6;
const int RightMotorForward = 3;
const int RightMotorBackward = 11;

// Ultrasonic sensor pins
#define trig_pin A3 // Trigger pin for ultrasonic sensor
#define echo_pin A2 // Echo pin for ultrasonic sensor
#define auto_pin A7
#define relay_pin A5
#define maximum_distance 200
int distance = 100;
int motorSpeed = 100;
int command;
int distanceRight = 0;
int distanceLeft = 0;
unsigned long lastCheckTime = 0;
const int checkInterval = 50; // 50ms interval for auto mode

NewPing sonar(trig_pin, echo_pin, maximum_distance); // Sensor function
Servo servo_motor; // Servo name
char lastCommand = 'S';  // Default is Stop

void setup() {
    Serial.begin(9600);
    pinMode(LeftMotorForward, OUTPUT);
    pinMode(LeftMotorBackward, OUTPUT);
    pinMode(RightMotorForward, OUTPUT);
    pinMode(RightMotorBackward, OUTPUT);
    pinMode(auto_pin, INPUT_PULLUP);
    pinMode(relay_pin, OUTPUT);

    servo_motor.attach(9); // Servo pin
    servo_motor.write(90);  // Center position for servo
    delay(2000);            // Wait for 2 seconds

    distance = readPing();
}

void loop() {
    controlRelay();
    int isAuto = analogRead(auto_pin);  // Check if in auto mode

    if (isAuto >= 900) {  // If Auto Mode is on
        if (millis() - lastCheckTime >= checkInterval) { // Use millis() instead of delay()
            lastCheckTime = millis();
            distance = readPing();

            if (distance <= 20) {  // If obstacle is detected within threshold
                moveStop();
                moveBackward(motorSpeed);
                delay(200);  // Move backward for a short time
                moveStop();
                //delay(500);  // Pause before navigating

                distanceRight = lookRight();  // Measure distance to the right
                distanceLeft = lookLeft();    // Measure distance to the left

                
                

                if (distanceRight < 20 && distanceLeft < 20) {  
                    turnRight(motorSpeed);  // Turn left if the right side is too close
                    delay(1200);    
                } 
                else if (distanceRight < 20) {  
                    turnRight(motorSpeed);  // Turn left if the right side is too close
                    delay(450);  
                } 
                else if (distanceLeft < 20) {  
                    turnLeft(motorSpeed);  // Turn right if the left side is too close
                    delay(450);  
                } 
                else if (distanceRight > distanceLeft) {  
                    turnLeft(motorSpeed);  // Turn left if the right side has more space
                    delay(450);  
                } 
                else {  
                    turnRight(motorSpeed);  // Turn right if the left side has more space
                    delay(450);  
                }

            } else {
                moveForward(motorSpeed);  // Keep moving forward if no obstacle
            }
        }
    } else {  // Manual mode
        if (Serial.available()) {
            lastCommand = Serial.read();  // Update last command
        }

        switch (lastCommand) {
            case 'F':
                moveForward(motorSpeed);
                break;
            case 'B':
                moveBackward(motorSpeed);
                break;
            case 'L':
                turnLeft(motorSpeed);
                break;
            case 'R':
                turnRight(motorSpeed);
                break;
            default:
                moveStop();  // Stop if no valid command
                break;
        }
    }
}

// Servo look right
// Servo look right and return to center
int lookRight() {
    servo_motor.write(10);  // Turn servo to the right
    delay(300);              // Wait for the servo to move
    int distance = readPing();  // Measure distance to the right
    servo_motor.write(90);  // Return servo to center position
    delay(300);              // Ensure it stabilizes at center
    return distance;
}

// Servo look left and return to center
int lookLeft() {
    servo_motor.write(170);  // Turn servo to the left
    delay(300);              // Wait for the servo to move
    int distance = readPing();  // Measure distance to the left
    servo_motor.write(90);  // Return servo to center position
    delay(300);              // Ensure it stabilizes at center
    return distance;
}


// Read distance from ultrasonic sensor
int readPing() {
    int cm = sonar.ping_cm();
    if (cm == 0) {
        cm = 250;  // Return max distance if no obstacle detected
    }
    return cm;
}

// Stop the robot
void moveStop() {
    analogWrite(LeftMotorForward, 0);
    analogWrite(RightMotorForward, 0);
    analogWrite(LeftMotorBackward, 0);
    analogWrite(RightMotorBackward, 0);
}

// Move forward with specified speed
void moveForward(int speed) {
    analogWrite(LeftMotorForward, speed);
    analogWrite(RightMotorForward, speed);
    analogWrite(LeftMotorBackward, 0);
    analogWrite(RightMotorBackward, 0);
}

// Move backward with specified speed
void moveBackward(int speed) {
    analogWrite(LeftMotorBackward, speed);
    analogWrite(RightMotorBackward, speed);
    analogWrite(LeftMotorForward, 0);
    analogWrite(RightMotorForward, 0);
}

// Turn right with specified speed
void turnRight(int speed) {
    analogWrite(LeftMotorForward, speed);
    analogWrite(RightMotorBackward, speed);
    analogWrite(LeftMotorBackward, 0);
    analogWrite(RightMotorForward, 0);
}

// Turn left with specified speed
void turnLeft(int speed) {
    analogWrite(LeftMotorBackward, speed);
    analogWrite(RightMotorForward, speed);
    analogWrite(LeftMotorForward, 0);
    analogWrite(RightMotorBackward, 0);
}
//void controlRelay() {
//    digitalWrite(relay_pin, LOW);
//    delay(3000);
//    digitalWrite(relay_pin, HIGH);
//    delay(3000);
//}
void controlRelay() {
    static unsigned long lastTime = 0;
    static bool relayState = LOW; 
    const unsigned long intervalOn = 3000;  // 3 seconds ON
    const unsigned long intervalOff = 2000; // 2 seconds OFF

    unsigned long currentTime = millis();
    
    if (relayState == LOW && (currentTime - lastTime >= intervalOn)) {
        relayState = HIGH;
        lastTime = currentTime;
        digitalWrite(relay_pin, HIGH);
    } 
    else if (relayState == HIGH && (currentTime - lastTime >= intervalOff)) {
        relayState = LOW;
        lastTime = currentTime;
        digitalWrite(relay_pin, LOW);
    }
}
