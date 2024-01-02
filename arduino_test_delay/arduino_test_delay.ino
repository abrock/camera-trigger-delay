#define ENERGY_WASTE 12

#define CAM_1_FOCUS 2
#define CAM_1_TRIGGER 3

#define CAM_2_FOCUS 4
#define CAM_2_TRIGGER 5

#define TEST_LED_R 8
#define TEST_LED_G 6
#define TEST_LED_B 7

uint32_t energy_waste_time;

void setup() {
  Serial.begin(9600);

  // Initialize focus and trigger lines
  for (int ii = 2; ii <= 5; ++ii) {
    pinMode(ii, OUTPUT);
    digitalWrite(ii, LOW);
  }
  // Initialize buttons
  for (int ii = 9; ii <= 10; ++ii) {
    pinMode(ii, INPUT_PULLUP);
  }
  // Initialize test-LEDs
  for (int ii = 6; ii <= 8; ++ii) {
    pinMode(ii, OUTPUT);
    digitalWrite(ii, HIGH);
  }

  pinMode(ENERGY_WASTE, OUTPUT);
  digitalWrite(ENERGY_WASTE, LOW);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

}

void trigger_high(bool cam1, bool cam2) {
  if (cam1) {
    digitalWrite(CAM_1_FOCUS, HIGH);
    digitalWrite(CAM_1_TRIGGER, HIGH);
  }
  if (cam2) {
    digitalWrite(CAM_2_FOCUS, HIGH);
    digitalWrite(CAM_2_TRIGGER, HIGH);
  }
}

void trigger_low() {
  digitalWrite(CAM_1_FOCUS, LOW);
  digitalWrite(CAM_1_TRIGGER, LOW);
  digitalWrite(CAM_2_FOCUS, LOW);
  digitalWrite(CAM_2_TRIGGER, LOW);
}

void trigger() {
    trigger_high(true, true);
    _delay_ms(200);
    trigger_low();
}


String msg;

void red(bool on) {
  digitalWrite(TEST_LED_R, on ? LOW : HIGH);
}
void green(bool on) {
  digitalWrite(TEST_LED_G, on ? LOW : HIGH);
}
void blue(bool on) {
  digitalWrite(TEST_LED_B, on ? LOW : HIGH);
}

void flash_leds() {
  // Order of colors:
  // RGB
  // 100 red
  // 110 yellow
  // 111 white
  // 101 pink
  // 001 blue
  // 011 turqoise
  // 010 green
  #define FLASH_DURATION 3
  red(true);
  _delay_ms(FLASH_DURATION);
  green(true);
  _delay_ms(FLASH_DURATION);
  blue(true);
  _delay_ms(FLASH_DURATION);
  green(false);
  _delay_ms(FLASH_DURATION);
  red(false);
  _delay_ms(FLASH_DURATION);
  green(true);
  _delay_ms(FLASH_DURATION);
  blue(false);

  _delay_ms(1000);

  digitalWrite(TEST_LED_R, HIGH);
  digitalWrite(TEST_LED_G, HIGH);
  digitalWrite(TEST_LED_B, HIGH);
}

void handle_msg() {
  Serial.print("Echo ");
  Serial.println(msg);
  if (msg == "h") {
    trigger_high(true, true);
    Serial.println("Trigger set to high");
  }
  else if (msg == "l") {
    trigger_low();
    Serial.println("Trigger set to low");
  }
  else if (msg == "t") {
    trigger_high(true, true);
    _delay_us(400);
    flash_leds();
    trigger_low();
    Serial.println("Triggered");
  }
  else if (msg.startsWith("t ")) {
    int count = msg.substring(2).toInt();
    trigger_high(true, true);
    for (int ii = 0; ii < count; ++ii) {
      //_delay_us(100);
      _delay_ms(1);
    }
    flash_leds();
    trigger_low();
    Serial.print("Triggered, count ");
    Serial.println(count);
  }
  else if (msg == "r") {
    digitalWrite(TEST_LED_R, !digitalRead(TEST_LED_R));
  }
  else if (msg == "g") {
    digitalWrite(TEST_LED_G, !digitalRead(TEST_LED_G));
  }
  else if (msg == "b") {
    digitalWrite(TEST_LED_B, !digitalRead(TEST_LED_B));
  }
  else if (msg == "f") {
    flash_leds();
  }
  else if (msg.startsWith("a")) {
    int count = msg.substring(2).toInt();
    for (int ii = 0; ii < count; ++ii) {
      trigger_high(true, true);
      //_delay_us(100);
      _delay_ms(60);
      flash_leds();
      trigger_low();
      Serial.print("Auto-trigger, #");
      Serial.print(ii+1);
      Serial.print(" out of ");
      Serial.println(count);
      _delay_ms(3000);
    }
    Serial.println("Done, ready for new commands");
  }

  msg = "";
}

void loop() {

  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handle_msg();
    }
    else {
      msg += c;
    }
  }

}
