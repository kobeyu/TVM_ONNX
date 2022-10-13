extern "C"{
  #include <stdio.h>
  #include <stdlib.h>
  #include <assert.h>
  #include <libgen.h>
//  #include "cnpy.h"
  #include "nms.h"
}
int result = 0 ;

int valid_cnt = 10;
const int incomingByte = 240 ;
byte serialBuffer[incomingByte];
int pin_data;
void *ptr = (void*)serialBuffer;
char buff[50];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600) ;
  result = nms((float*)ptr, valid_cnt);
  pinMode(13, OUTPUT); // LED3 = 13
  Serial.flush() ;

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()){
    
      Serial.readBytes(serialBuffer, incomingByte);
      
      pin_data = nms((float*)ptr, valid_cnt);
      digitalWrite(pin_data,HIGH);
      
      Serial.write( (byte*)ptr, incomingByte);
      delay(5000);
      digitalWrite(13,LOW);

  }
}
