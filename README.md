# Experimentation

## 1. Initial Convolutional Neural Network (CNN)
To begin with, I used the following specs on the smaller dataset:

- KERNEL_SIZE = (3,3) 
- FILTERS = 32 
- POOL_SIZE = (2,2)
- DROPOUT_NEURONS = 128
- DROPOUT = 0.5 
- OUTPUT_NEURONS = 3

The resulting output '11/11 - 0s - 18ms/step - accuracy: 0.9643 - loss: 0.2094' shows a significant loss of 21%. Next, I'll run this on the larger dataset and see how this affects loss. Of course, the more logical thing to do would be to optimise the parameters before running on a larger dataset, but I'm intrigued to see the sigificance of more data on loss.

## 2. Same parameters, more data
With the same parameters that resulted in 21% loss when trained on the small dataset, I tried training on the large dataset. However, after a 20 minute run-time with 0 epochs run, I decided some optimisation would most likely be required.

## 3. Optimising parameters: less filters
Reduced filters from 32 -> 8. 

- FILTERS = 8

Output: 11/11 - 0s - 17ms/step - accuracy: 0.9970 - loss: 0.0925
Conclusion: Loss decreased, 1ms less per step!

## 4. Optimising parameters: filters
- FILTERS = 16

Output: 11/11 - 0s - 17ms/step - accuracy: 1.0000 - loss: 1.1672e-04
Conclusion: Loss decreased

## 5. Optimising parameters: DROPOUT_NEURONS
Reducing no. dropout neurons

- KERNEL_SIZE = (3,3) 
- FILTERS = 32 
- POOL_SIZE = (2,2)
- DROPOUT_NEURONS = 64
- DROPOUT = 0.1
- OUTPUT_NEURONS = 3

Output: 11/11 - 0s - 18ms/step - accuracy: 0.9940 - loss: 0.0333
Conclusion: Increase in loss

## 5. Optimising parameters: DROPOUT_NEURONS
Increase no. dropout neurons

- KERNEL_SIZE = (3,3) 
- FILTERS = 32 
- POOL_SIZE = (2,2)
- DROPOUT_NEURONS = 256
- DROPOUT = 0.1
- OUTPUT_NEURONS = 3

Conclusion: Decrease in loss, but doubled runtime.

## 6. Optimising parameters: DROPOUT
Increase dropout to 80%

- KERNEL_SIZE = (3,3) 
- FILTERS = 32 
- POOL_SIZE = (2,2)
- DROPOUT_NEURONS = 128
- DROPOUT = 0.8
- OUTPUT_NEURONS = 3

Output: 11/11 - 0s - 20ms/step - accuracy: 0.9970 - loss: 0.0332
Conclusion: Greater loss

## 6. Optimising parameters: DROPOUT
Decrease dropout to 10%

- KERNEL_SIZE = (3,3) 
- FILTERS = 32 
- POOL_SIZE = (2,2)
- DROPOUT_NEURONS = 128
- DROPOUT = 0.1
- OUTPUT_NEURONS = 3

Output: 11/11 - 0s - 19ms/step - accuracy: 1.0000 - loss: 1.0644e-09
Conclusion: Reduced loss significantly!

## 6. Optimising parameters: KERNEL_SIZE= (5,5)
Increase kernel_size

- KERNEL_SIZE = (5,5)

Observation: Increased runtime, reduced loss.

## 6. Optimising parameters: POOL_SIZE
Increase 

- POOL_SIZE = (4,4)

Observation: Increased runtime, decreased loss

