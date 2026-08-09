// =====================================================
// SMART RECYCLE BIN - Arduino Mega Controller
// Updated:
// - 4 bin-fullness HC-SR04 sensors on pins 46-53
// - FS5109R continuous rotation servo on pin 44
// - Servo does NOT attach during startup
// - Servo calibrated:
//      1600 us for 320 ms = unlock
//      1400 us for 300 ms = lock
//      Reverse brake pulse added after each servo spin
//      1500 us            = stop
// - Floor stepper always enabled/locked
// - Sends FULLNESS data over USB Serial to Raspberry Pi
// - Bin fullness now uses 10-sample averaging per sensor
// =====================================================

#include <Servo.h>

// =====================================================
// PIN DEFINITIONS
// =====================================================

// FLOOR STEPPER
#define FLOOR_EN    13
#define FLOOR_STEP  12
#define FLOOR_DIR   11

// LINEAR GUIDE STEPPER
#define LINEAR_EN    7
#define LINEAR_STEP  6
#define LINEAR_DIR   5

// MAIN DOOR STEPPER
#define DOOR_EN    4
#define DOOR_STEP  3
#define DOOR_DIR   2

// ENTRY HC-SR04
#define US_TRIG  9
#define US_ECHO  10

// SERVO FLOOR LOCK
#define SERVO_PIN 44

Servo gateServo;
bool servoAttached = false;

// Calibrated servo values from your test
int SERVO_STOP_US   = 1500;
int SERVO_UNLOCK_US = 1600;
int SERVO_LOCK_US   = 1400;

int SERVO_UNLOCK_TIME_MS = 320;
int SERVO_LOCK_TIME_MS   = 300;

// Brake = spin opposite direction for short time after main spin
int SERVO_UNLOCK_BRAKE_US = 1400;
int SERVO_UNLOCK_BRAKE_TIME_MS = 40;

int SERVO_LOCK_BRAKE_US = 1600;
int SERVO_LOCK_BRAKE_TIME_MS = 50;

int SERVO_FORCE_STOP_TIME_MS = 500;

// PI COMMUNICATION
#define UI_DOOR_EVENT_PIN A4
#define PI_TRIGGER_PIN 8

// Pi -> Arduino result pins
#define GLASS_PIN    A0
#define METAL_PIN    A1
#define PAPER_PIN    A2
#define PLASTIC_PIN  A3

// BIN FULLNESS HC-SR04 SENSORS
// calibrated mapping:
// metal   echo 46, trig 47
// glass   echo 52, trig 53
// paper   echo 50, trig 51
// plastic echo 48, trig 49
const int BIN_COUNT = 4;

const int BIN_ECHO[BIN_COUNT] = {46, 52, 50, 48};
const int BIN_TRIG[BIN_COUNT] = {47, 53, 51, 49};

const char* BIN_NAME[BIN_COUNT] = {
  "metal",
  "glass",
  "paper",
  "plastic"
};

// Order: metal, glass, paper, plastic
// Calibrated values are based on: FULL distance, EMPTY distance
// metal   full 4.27 cm,  empty 32.58 cm
// glass   full 5.33 cm,  empty 33.07 cm
// paper   full 5.20 cm,  empty 32.12 cm
// plastic full 22.16 cm, empty 30.94 cm
float EMPTY_DISTANCE_CM[BIN_COUNT] = {32.58, 33.07, 32.12, 30.94};
float FULL_DISTANCE_CM[BIN_COUNT]  = {4.27, 5.33, 5.20, 22.16};

int latestFullness[BIN_COUNT] = {0, 0, 0, 0};
bool operatorDisabled = false;

// =====================================================
// PARAMETERS
// =====================================================

int floor_speed_us  = 1000;
int linear_speed_us = 10000;
int door_speed_us   = 3000;

int floor_move_steps = 950;
int door_open_steps  = 85;

// Calibrated chamber linear movement steps
// Glass and metal move forward from centre.
// Paper and plastic move backward from centre.
int GLASS_TO_STEPS     = 70;
int GLASS_BACK_STEPS   = 75;

int METAL_TO_STEPS     = 165;
int METAL_BACK_STEPS   = 162;

int PAPER_TO_STEPS     = 60;
int PAPER_BACK_STEPS   = 57;

int PLASTIC_TO_STEPS   = 180;
int PLASTIC_BACK_STEPS = 177;

float trigger_distance_cm = 50.0;
int required_consecutive_reads = 3;

// Door anti-close safety
// Physical door will not close while something/person is within 30 cm.
float door_safety_distance_cm = 30.0;
unsigned long door_min_open_time_ms = 5000;
int door_clear_required_reads = 10;

unsigned long pi_result_timeout_ms = 60000;

// =====================================================
// USB SERIAL COMMANDS FROM RASPBERRY PI
// =====================================================

void sendBinFullnessToPi();
bool allBinsFull();

void handleSerialCommands()
{
  while (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();
    if (cmd.length() == 0) return;
    if (cmd == "DISABLE")
    {
      operatorDisabled = true;
      Serial.println("COMMAND_ACK,DISABLE");
    }
    else if (cmd == "ENABLE")
    {
      operatorDisabled = false;
      Serial.println("COMMAND_ACK,ENABLE");
    }
    else if (cmd == "REQUEST_FULLNESS")
    {
      Serial.println("COMMAND_ACK,REQUEST_FULLNESS");
      sendBinFullnessToPi();
    }
    else if (cmd == "STATUS")
    {
      Serial.print("STATUS,operatorDisabled=");
      Serial.print(operatorDisabled ? 1 : 0);
      Serial.print(",allBinsFull=");
      Serial.println(allBinsFull() ? 1 : 0);
    }
    else
    {
      Serial.print("COMMAND_UNKNOWN,");
      Serial.println(cmd);
    }
  }
}

// =====================================================
// PULSE HELPERS
// =====================================================

void pulsePin(int pin, int pulseMs)
{
  digitalWrite(pin, HIGH);
  delay(pulseMs);
  digitalWrite(pin, LOW);
}

void pulseUiDoorEvent()
{
  Serial.println("Pulsing UI door/open event to Pi");
  pulsePin(UI_DOOR_EVENT_PIN, 500);
}

void pulsePiClassificationTrigger()
{
  Serial.println("Triggering Pi classification");
  pulsePin(PI_TRIGGER_PIN, 500);
}

// =====================================================
// STEPPER FUNCTIONS
// =====================================================

void moveStepperNormal(int enPin, int stepPin, int dirPin, bool direction, int steps, int pulseDelay)
{
  digitalWrite(enPin, LOW);
  digitalWrite(dirPin, direction);

  for (int i = 0; i < steps; i++)
  {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(pulseDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(pulseDelay);
  }

  digitalWrite(enPin, HIGH);
}

// Floor stepper stays enabled/locked
void moveFloorStepper(bool direction, int steps, int pulseDelay)
{
  digitalWrite(FLOOR_EN, LOW);
  digitalWrite(FLOOR_DIR, direction);

  for (int i = 0; i < steps; i++)
  {
    digitalWrite(FLOOR_STEP, HIGH);
    delayMicroseconds(pulseDelay);
    digitalWrite(FLOOR_STEP, LOW);
    delayMicroseconds(pulseDelay);
  }

  digitalWrite(FLOOR_EN, LOW);
}

// =====================================================
// SERVO LOCK FUNCTIONS
// =====================================================

void servoAttachIfNeeded()
{
  if (!servoAttached)
  {
    gateServo.attach(SERVO_PIN);
    servoAttached = true;
    delay(80);
  }
}

void servoStop()
{
  if (servoAttached)
  {
    gateServo.writeMicroseconds(SERVO_STOP_US);
    delay(SERVO_FORCE_STOP_TIME_MS);

    gateServo.writeMicroseconds(SERVO_STOP_US);
    delay(SERVO_FORCE_STOP_TIME_MS);
  }
}

void servoDetach()
{
  if (servoAttached)
  {
    gateServo.detach();
    servoAttached = false;
  }
}

// +90 degree unlock
void servoUnlockFloor()
{
  Serial.println("Servo floor lock: UNLOCK");

  servoAttachIfNeeded();

  gateServo.writeMicroseconds(SERVO_UNLOCK_US);
  delay(SERVO_UNLOCK_TIME_MS);

  gateServo.writeMicroseconds(SERVO_UNLOCK_BRAKE_US);
  delay(SERVO_UNLOCK_BRAKE_TIME_MS);

  servoStop();
  servoDetach();
}

// -90 degree lock
void servoLockFloor()
{
  Serial.println("Servo floor lock: LOCK");

  servoAttachIfNeeded();

  gateServo.writeMicroseconds(SERVO_LOCK_US);
  delay(SERVO_LOCK_TIME_MS);

  gateServo.writeMicroseconds(SERVO_LOCK_BRAKE_US);
  delay(SERVO_LOCK_BRAKE_TIME_MS);

  servoStop();
  servoDetach();
}

// Keep old function names so the rest of the code still works
void servoRotateForward90()
{
  servoUnlockFloor();
}

void servoRotateBackward90()
{
  servoLockFloor();
}

// =====================================================
// FLOOR FUNCTIONS
// =====================================================

void floor_open()
{
  Serial.println("Opening floor");
  moveFloorStepper(HIGH, floor_move_steps, floor_speed_us);
}

void floor_close()
{
  Serial.println("Closing floor");
  moveFloorStepper(LOW, floor_move_steps, floor_speed_us);
}

void dump_item_with_servo_and_floor()
{
  // 1. Servo unlocks the seesaw floor
  // 2. Floor opens
  // 3. Floor closes
  // 4. Servo locks the floor again

  servoUnlockFloor();
  delay(150);

  floor_open();
  delay(888);

  floor_close();
  delay(150);

  servoLockFloor();
  delay(150);

  digitalWrite(FLOOR_EN, LOW);
}

// =====================================================
// DOOR FUNCTIONS
// =====================================================

void door_open()
{
  Serial.println("Opening door");
  moveStepperNormal(DOOR_EN, DOOR_STEP, DOOR_DIR, HIGH, door_open_steps, door_speed_us);
}

void door_close()
{
  Serial.println("Closing door");
  moveStepperNormal(DOOR_EN, DOOR_STEP, DOOR_DIR, LOW, door_open_steps, door_speed_us);
}

// =====================================================
// LINEAR GUIDE FUNCTIONS
// =====================================================

void move_linear_forward_steps(int steps)
{
  Serial.print("Moving forward ");
  Serial.print(steps);
  Serial.println(" steps");

  moveStepperNormal(LINEAR_EN, LINEAR_STEP, LINEAR_DIR, HIGH, steps, linear_speed_us);
}

void move_linear_backward_steps(int steps)
{
  Serial.print("Moving backward ");
  Serial.print(steps);
  Serial.println(" steps");

  moveStepperNormal(LINEAR_EN, LINEAR_STEP, LINEAR_DIR, LOW, steps, linear_speed_us);
}

// =====================================================
// ULTRASONIC FUNCTIONS
// =====================================================

float readEntryDistanceCM()
{
  digitalWrite(US_TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(US_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(US_TRIG, LOW);

  long duration = pulseIn(US_ECHO, HIGH, 30000);

  if (duration == 0)
  {
    return 999.0;
  }

  return duration * 0.0343 / 2.0;
}


bool doorAreaIsOccupied()
{
  float d = readEntryDistanceCM();
  return (d > 0.0 && d < door_safety_distance_cm);
}

void waitForDoorAreaClearBeforeClosing()
{
  unsigned long startTime = millis();

  // Keep the door open for the normal insert time first.
  while (millis() - startTime < door_min_open_time_ms)
  {
    handleSerialCommands();
    delay(20);
  }

  // After the normal open time, only close when the area is clear.
  int clearReads = 0;

  while (clearReads < door_clear_required_reads)
  {
    handleSerialCommands();

    if (doorAreaIsOccupied())
    {
      clearReads = 0;
      Serial.println("DOOR_SAFETY,HOLD_OPEN,person_within_30cm");
    }
    else
    {
      clearReads++;
      Serial.print("DOOR_SAFETY,CLEAR_READS=");
      Serial.println(clearReads);
    }

    delay(80);
  }

  Serial.println("DOOR_SAFETY,CLEAR_TO_CLOSE");
}

float readDistanceCMFromPins(int trigPin, int echoPin)
{
  digitalWrite(trigPin, LOW);
  delayMicroseconds(3);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, 30000UL);

  if (duration == 0)
  {
    return -1.0;
  }

  return duration * 0.0343 / 2.0;
}

float readAverageDistanceCM(int trigPin, int echoPin)
{
  const int SAMPLE_COUNT = 10;
  float sum = 0.0;
  int validCount = 0;

  for (int sample = 0; sample < SAMPLE_COUNT; sample++)
  {
    float d = readDistanceCMFromPins(trigPin, echoPin);

    // HC-SR04 practical valid range. Invalid timeout remains -1.0.
    if (d > 0.0 && d < 400.0)
    {
      sum += d;
      validCount++;
    }

    delay(40);
  }

  if (validCount == 0)
  {
    return -1.0;
  }

  return sum / validCount;
}


int distanceToFullnessPercent(float distanceCM, float emptyCM, float fullCM)
{
  if (distanceCM < 0)
  {
    return -1;
  }

  float percent = (emptyCM - distanceCM) / (emptyCM - fullCM) * 100.0;

  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;

  return (int)(percent + 0.5);
}

void sendBinFullnessToPi()
{
  int fullness[BIN_COUNT];
  float distance[BIN_COUNT];

  Serial.println("Reading bin fullness sensors...");

  for (int i = 0; i < BIN_COUNT; i++)
  {
    distance[i] = readAverageDistanceCM(BIN_TRIG[i], BIN_ECHO[i]);

    fullness[i] = distanceToFullnessPercent(
      distance[i],
      EMPTY_DISTANCE_CM[i],
      FULL_DISTANCE_CM[i]
    );

    latestFullness[i] = fullness[i];

    delay(70);
  }

  Serial.print("FULLNESS,");
  Serial.print("metal=");
  Serial.print(fullness[0]);
  Serial.print(",glass=");
  Serial.print(fullness[1]);
  Serial.print(",paper=");
  Serial.print(fullness[2]);
  Serial.print(",plastic=");
  Serial.println(fullness[3]);

  Serial.print("Distances cm: metal=");
  Serial.print(distance[0]);
  Serial.print(", glass=");
  Serial.print(distance[1]);
  Serial.print(", paper=");
  Serial.print(distance[2]);
  Serial.print(", plastic=");
  Serial.println(distance[3]);
}

bool allBinsFull()
{
  for (int i = 0; i < BIN_COUNT; i++)
  {
    // If sensor fault is -1, do not treat it as full. Laptop will alert operator.
    if (latestFullness[i] < 100)
    {
      return false;
    }
  }
  return true;
}

// =====================================================
// WAIT FOR PI CLASSIFICATION
// Return:
// 0 = null / no sort / timeout
// 1 = glass
// 2 = metal
// 3 = paper
// 4 = plastic
// =====================================================

int waitForPiClassification()
{
  Serial.println("Waiting for Pi classification...");

  while (
    digitalRead(GLASS_PIN) == HIGH ||
    digitalRead(METAL_PIN) == HIGH ||
    digitalRead(PAPER_PIN) == HIGH ||
    digitalRead(PLASTIC_PIN) == HIGH
  )
  {
    Serial.println("Waiting for result pins to reset LOW...");
    delay(50);
  }

  Serial.println("Result pins LOW. Waiting for Pi HIGH pulse...");

  unsigned long startTime = millis();

  while (true)
  {
    handleSerialCommands();
    bool glassHigh = digitalRead(GLASS_PIN) == HIGH;
    bool metalHigh = digitalRead(METAL_PIN) == HIGH;
    bool paperHigh = digitalRead(PAPER_PIN) == HIGH;
    bool plasticHigh = digitalRead(PLASTIC_PIN) == HIGH;

    if (glassHigh && metalHigh && paperHigh && plasticHigh)
    {
      Serial.println("Detected: NULL / NO SORT");
      return 0;
    }

    if (glassHigh)
    {
      Serial.println("Detected: GLASS");
      return 1;
    }

    if (metalHigh)
    {
      Serial.println("Detected: METAL");
      return 2;
    }

    if (paperHigh)
    {
      Serial.println("Detected: PAPER");
      return 3;
    }

    if (plasticHigh)
    {
      Serial.println("Detected: PLASTIC");
      return 4;
    }

    if (millis() - startTime > pi_result_timeout_ms)
    {
      Serial.println("Pi classification timeout. No sort.");
      return 0;
    }

    delay(10);
  }
}

// =====================================================
// SORTING SEQUENCE
// =====================================================

void executeSorting(int material)
{
  if (material == 0)
  {
    Serial.println("No valid recyclable class. Skipping sorting.");
    return;
  }

  if (material == 1) // GLASS = forward from centre
  {
    move_linear_forward_steps(GLASS_TO_STEPS);
   

    dump_item_with_servo_and_floor();

    move_linear_backward_steps(GLASS_BACK_STEPS);
    
  }

  else if (material == 2) // METAL = forward from centre
  {
    move_linear_forward_steps(METAL_TO_STEPS);
    

    dump_item_with_servo_and_floor();

    move_linear_backward_steps(METAL_BACK_STEPS);
   
  }

  else if (material == 3) // PAPER = backward from centre
  {
    move_linear_backward_steps(PAPER_TO_STEPS);
    

    dump_item_with_servo_and_floor();

    move_linear_forward_steps(PAPER_BACK_STEPS);
   
  }

  else if (material == 4) // PLASTIC = backward from centre
  {
    move_linear_backward_steps(PLASTIC_TO_STEPS);
   

    dump_item_with_servo_and_floor();

    move_linear_forward_steps(PLASTIC_BACK_STEPS);
 
  }

  digitalWrite(FLOOR_EN, LOW);
}

// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(9600);
  Serial.setTimeout(50);

  pinMode(FLOOR_EN, OUTPUT);
  pinMode(FLOOR_STEP, OUTPUT);
  pinMode(FLOOR_DIR, OUTPUT);

  pinMode(LINEAR_EN, OUTPUT);
  pinMode(LINEAR_STEP, OUTPUT);
  pinMode(LINEAR_DIR, OUTPUT);

  pinMode(DOOR_EN, OUTPUT);
  pinMode(DOOR_STEP, OUTPUT);
  pinMode(DOOR_DIR, OUTPUT);

  pinMode(US_TRIG, OUTPUT);
  pinMode(US_ECHO, INPUT);

  for (int i = 0; i < BIN_COUNT; i++)
  {
    pinMode(BIN_TRIG[i], OUTPUT);
    pinMode(BIN_ECHO[i], INPUT);
    digitalWrite(BIN_TRIG[i], LOW);
  }

  pinMode(PI_TRIGGER_PIN, OUTPUT);
  digitalWrite(PI_TRIGGER_PIN, LOW);

  pinMode(UI_DOOR_EVENT_PIN, OUTPUT);
  digitalWrite(UI_DOOR_EVENT_PIN, LOW);

  pinMode(GLASS_PIN, INPUT);
  pinMode(METAL_PIN, INPUT);
  pinMode(PAPER_PIN, INPUT);
  pinMode(PLASTIC_PIN, INPUT);

  digitalWrite(FLOOR_EN, LOW);
  digitalWrite(LINEAR_EN, HIGH);
  digitalWrite(DOOR_EN, HIGH);

  // Servo intentionally NOT attached on startup
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);

  delay(1500);

  Serial.println("=================================");
  Serial.println("SMART RECYCLE BIN STARTED");
  Serial.println("Floor stepper is ENABLED/LOCKED");
  Serial.println("Servo lock is NOT attached on startup");
  Serial.println("Operator remote disable is supported through USB Serial");
  Serial.println("Calibrated bin fullness and linear movement loaded");
  Serial.println("=================================");

  sendBinFullnessToPi();
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  digitalWrite(FLOOR_EN, LOW);

  Serial.println("\nWaiting for person...");

  // Service lockout: if all bins are full, do not open the door.
  // Keep sending fullness to Pi so the laptop UI can show Out of Service.
  handleSerialCommands();
  sendBinFullnessToPi();
  handleSerialCommands();

  if (operatorDisabled)
  {
    Serial.println("OPERATOR DISABLED. Door locked out until remote enable.");
    delay(2000);
    return;
  }

  if (allBinsFull())
  {
    Serial.println("ALL BINS FULL. Door locked out until bins are serviced.");
    delay(2000);
    return;
  }

  int consecutiveDetects = 0;

  while (true)
  {
    digitalWrite(FLOOR_EN, LOW);
    handleSerialCommands();
    if (operatorDisabled)
    {
      Serial.println("OPERATOR DISABLED during wait. Door will not open.");
      delay(1000);
      return;
    }

    float distance = readEntryDistanceCM();

    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    if (distance < trigger_distance_cm)
    {
      consecutiveDetects++;

      Serial.print("Consecutive detects: ");
      Serial.println(consecutiveDetects);

      if (consecutiveDetects >= required_consecutive_reads)
      {
        Serial.println("Person confirmed");
        break;
      }
    }
    else
    {
      consecutiveDetects = 0;
    }

    delay(50);
  }

  // Re-check just before opening in case bins became full while waiting.
  handleSerialCommands();
  if (operatorDisabled)
  {
    Serial.println("OPERATOR DISABLED after user detection. Door will not open.");
    delay(2000);
    return;
  }

  if (allBinsFull())
  {
    Serial.println("ALL BINS FULL after user detection. Door will not open.");
    delay(2000);
    return;
  }

  pulseUiDoorEvent();

  door_open();
  waitForDoorAreaClearBeforeClosing();
  door_close();

  pulsePiClassificationTrigger();
  sendBinFullnessToPi();
  int result = waitForPiClassification();

  executeSorting(result);

  handleSerialCommands();

  digitalWrite(FLOOR_EN, LOW);

  delay(1000);
}