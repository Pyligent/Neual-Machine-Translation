# Neual-Machine-Translation
### Bidirectional LSTM with Multiplicative Attention   

### 1. Summary
Neural Machine Translation is to convert a sentence from the source language (e.g. French) to the target language (e.g. English). In this assignment, we will implement a sequence-to-sequence (Seq2Seq) network with attention, to build a Neural Machine Translation (NMT) system. In this section, we describe the training procedure for the proposed NMT system, which uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.


### 2. Network Architecture
![img1](./images/nmt.png)

### 3. Model
- Encoder : Hidden State and Cell State
![img2](./images/hc.png)

- Decoder : Hidden State and Cell State
  - Initialize the Decoder's first hidden state and Cell State. (Bidirectional: shape is 2*hidden_size)
  ![img3](./images/de.png)
  
  - On time-stamp t, Decoder's Hidden State and Cell State
  ![img4](./images/de1.png)
  
- Multiplicative Attention
![img5](./images/mul.png)

- Output Vector 
concatenate the attention output at with the decoder hidden state hdect and pass this through a linear layer, Tanh, and Dropout to attain the combined-output vector ot.  
![img6](./images/ot.png)

- Probability distribution Pt over target words at the time-stamp t
![img7](./images/pt.png)
