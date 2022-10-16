extern "C"{
  #include <stdio.h>
  #include <stdlib.h>
  #include <assert.h>
  #include <libgen.h>
//  #include "cnpy.h"
  #include "nms.h"
  #include <nds_intrinsic.h>
}
int result = 0 ;

int valid_cnt = 5;
const int incomingByte = 120 ;
byte serialBuffer[incomingByte];
int pin_data;
void *ptr = (void*)serialBuffer;
char buff[50];
unsigned long time_start;
unsigned long time_end;

int itr = 1000000;



void setup() {
  // put your setup code here, to run once:
  Serial.begin(38400) ;
  result = nms((float*)ptr, valid_cnt);
  pinMode(13, OUTPUT); // LED3 = 13
  Serial.flush() ;
  

}


void loop() {
 // put your main code here, to run repeatedly:
 if (Serial.available()){
   
     Serial.readBytes(serialBuffer, incomingByte);
     pin_data = nms((float*)ptr, valid_cnt);
     Serial.write( (byte*)ptr, incomingByte);
 }
 
}



