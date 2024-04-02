/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/


#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <iostream>
#include "data.h"
#include "model.h"



// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 32 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* DIGITS[] = {
"1", "2", "3", "4", "5", "6", "7", "8","9"
};

#define NUM_DIGITS (sizeof(DIGITS) / sizeof(DIGITS[0]))

unsigned long finalmodelStartTime, finalmodelEndTime;
unsigned long avgFinalmodelTime;
unsigned int count = 0;
float result = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model_tflite);  
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  unsigned int i;

      for (size_t i = 0; i < sizeof(input_data) / sizeof(float); i++) {
      tflInputTensor->data.f[i] = input_data[i];
      }
  
      count+=1;
      // Run inferencing
      finalmodelStartTime = micros();
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!");
        while (1);
        return;
      }
      finalmodelEndTime = micros();
      Serial.print("[log] Runtime of finalmodel: ");
      Serial.print(finalmodelEndTime - finalmodelStartTime);
      Serial.println(" microseconds");
      avgFinalmodelTime += finalmodelEndTime - finalmodelStartTime;
      result = (float)avgFinalmodelTime / count;
      Serial.print("[log] AVG Runtime of finalmodel: ");
      Serial.print(result);
      Serial.println(" microseconds");
              
      // Loop through the output tensor values from the model
      for (i = 0; i < NUM_DIGITS; i++) {
        Serial.print(DIGITS[i]);
        Serial.print(": ");
        
         Serial.println(tflOutputTensor->data.f[i], 6);

      }
      Serial.println();
      
    
  
}
