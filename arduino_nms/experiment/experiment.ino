extern "C"{
  #include <stdio.h>
  #include <stdlib.h>
  #include <assert.h>
  #include <nds_intrinsic.h>
}
int result = 0 ;

unsigned long time_start;
unsigned long time_end;
int itr = 1000000;



void setup() {
  // put your setup code here, to run once:
  Serial.begin(38400) ;
  Serial.flush() ;
}
void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()){
    Serial.read();
    time_start = millis();
    Serial.println( "Test SIMD cmpeq" );
    nds_cmpeq();
    Serial.println( "Test Original time" );
    cmpeq();
    Serial.println( "Test SIMD sub8" );
    nds_sub8();
    Serial.println( "Test Original time" );
    sub();
    Serial.println( "Done" );
    
  }
  
}

void nds_cmpeq(){
   uint32_t data = 0X11223344;
   uint32_t threshold = 0X22222222;
   uint32_t result = 0; 
   int i = 0;
   // Start timer
   time_start = millis();
   for (i=0 ; i<itr; i++){
      result = __nds__scmplt8(data, threshold);
   }
   time_end = millis();
   
   
   Serial.println( "Test SIMD time" );
   Serial.println( time_end - time_start );
   Serial.println( "Result" );
   Serial.println( result );
}
//
void cmpeq(){
   uint8_t data[4] = {0x11, 0x22, 0x33, 0x44};
   uint8_t threshold=0X22;
   uint8_t result[4] = {0,0,0,0}; 
   int i , j;
   // Start timer
   time_start = millis();
   for (i=0; i<itr; i++){
   for( j=0; j<4; j++){
      if(data[j] < threshold)
        result[j] = 0xff;   
      else
        result[j] = 0X00;
      }
   }
   time_end = millis();
   Serial.println( "Test Normal time" );
//   Serial.println( time_end );
   Serial.println( time_end - time_start );
   uint32_t tmp = (result[0]<<24) | (result[1]<<16) | (result[2]<<8) | (result[3]) ;
   Serial.println( "Result" );
   Serial.println( tmp );
}

void nds_sub8(){
   uint32_t data = 0X11223344;
   uint32_t sub_data =0X11111111;
   uint32_t result = 0; 
   int i = 0;
   // Start timer
   time_start = millis();
   for (i=0 ; i<itr; i++){
      result = __nds__sub8(data, sub_data);
   }
   time_end = millis();
   
   
   Serial.println( "Test SIMD time" );
   Serial.println( time_end - time_start );
   Serial.println( "Result" );
   Serial.println( result );
}

void sub(){
   uint8_t data[4] = {0x11, 0x22, 0x33, 0x44};
   uint8_t sub_data= 0x11;
   uint8_t result[4] = {0,0,0,0}; 
   int i , j;
   // Start timer
   time_start = millis();
   for (i=0; i<itr; i++){
   for( j=0; j<4; j++){
        result[j] = data[j] - sub_data;   
      }
   }
   time_end = millis();
   Serial.println( "Test Normal time" );
//   Serial.println( time_end );
   Serial.println( time_end - time_start );
   uint32_t tmp = (result[0]<<24) | (result[1]<<16) | (result[2]<<8) | (result[3]) ;
   Serial.println( "Result" );
   Serial.println( tmp );
}
