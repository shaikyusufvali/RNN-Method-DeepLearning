# 🧠 RNN – Deep Learning | Shaik Yusuf Vali

<p align="center">
  <img src="https://img.shields.io/badge/Python-Programming-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-RNN-purple?style=for-the-badge" alt="Deep Learning RNN"/>
  <img src="https://img.shields.io/badge/NumPy-Mathematics-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Status-Learning%20%26%20Development-orange?style=for-the-badge" alt="Project Status"/>
</p>

## 👋 About Me

Hi, I'm **Shaik Yusuf Vali**, a B.Tech student specializing in Artificial Intelligence and Machine Learning.

I'm interested in understanding how Deep Learning models work internally, from mathematical foundations to practical Python implementations.

This repository documents my learning journey with **Recurrent Neural Networks (RNNs)**, including mathematical calculations, forward propagation, hidden states, token embeddings, output layers, next-token prediction, and Backpropagation Through Time.

My goal is to understand the concepts step by step and implement them using Python and Deep Learning libraries.

## 📌 About This Repository

**RNN-Deep-Learning** is a learning and practice repository focused on understanding Recurrent Neural Networks from fundamentals to practical implementation.

RNNs are neural networks designed to process sequential data. They maintain a hidden state that carries information from previous time steps, allowing the network to use context from earlier inputs.

This repository focuses on understanding the mathematics behind RNNs instead of treating them as black-box models.

The learning process includes:

- Understanding concepts with simple examples.
- Performing mathematical calculations manually.
- Implementing calculations using Python and NumPy.
- Understanding how data flows through an RNN.
- Exploring how neural networks learn from prediction errors.
- Building small implementations and experiments.

## 🎯 Learning Objectives

- Understand the fundamentals of Deep Learning.
- Learn the architecture of Recurrent Neural Networks.
- Understand sequential data and time steps.
- Learn tokenization and token-to-ID mapping.
- Understand embedding matrices and embedding vectors.
- Learn input-to-hidden and hidden-to-hidden connections.
- Calculate hidden states step by step.
- Understand weights, biases, and activation functions.
- Learn matrix multiplication in neural networks.
- Understand output layers and logits.
- Learn Softmax and probability distributions.
- Understand next-token prediction.
- Learn loss functions and prediction errors.
- Understand Backpropagation Through Time (BPTT).
- Explore vanishing and exploding gradients.
- Implement simple RNN models using Python.
- Explore LSTM and GRU architectures.
- Apply RNNs to Natural Language Processing and time-series problems.

## 📚 Topics Covered

### 1. RNN Fundamentals

- Introduction to Deep Learning.
- What is a Recurrent Neural Network?
- Why do we need RNNs?
- Feedforward Neural Networks vs RNNs.
- Sequential data.
- Time steps.
- RNN architecture.
- Hidden states.
- Context and sequence memory.
- Real-world applications of RNNs.

### 2. Tokenization and Embeddings

- Text preprocessing.
- Tokenization.
- Vocabulary creation.
- Token-to-ID mapping.
- Embedding matrices.
- Embedding vectors.
- Word representations.
- How embeddings are used as model inputs.
- How embedding parameters can be learned during training.

Example:

Input text:

"I love India"

The text can be divided into tokens:

"I", "love", "India"

Each token is mapped to a token ID using a vocabulary. The token ID is then used to retrieve its corresponding embedding vector.

The embedding vector is passed to the neural network as a numerical representation of the token.

### 3. RNN Forward Propagation

Forward propagation describes how input information moves through an RNN to calculate hidden states and outputs.

Topics include:

- Input vectors.
- Previous hidden states.
- Input-to-hidden weights.
- Hidden-to-hidden weights.
- Bias terms.
- Activation functions.
- Hidden-state calculations.
- Sequential processing.
- Shared weights across time steps.

Basic hidden-state equation:

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

Where:

- h_t = Current hidden state.
- h_(t-1) = Previous hidden state.
- x_t = Current input vector.
- W_xh = Input-to-hidden weight matrix.
- W_hh = Hidden-to-hidden weight matrix.
- b_h = Hidden-layer bias.
- tanh = Hyperbolic tangent activation function.

The previous hidden state carries information from earlier time steps into the current calculation.

### 4. Hidden States and Sequential Memory

- Meaning of a hidden state.
- How information moves across time steps.
- Initial hidden state.
- Hidden-state updates.
- Context representation.
- How previous inputs influence later outputs.
- Hidden-state dimensions.
- Numerical hidden-state calculations.

A hidden state is a learned numerical representation of information from the sequence processed so far. It is not itself a word or a complete human-readable sentence.

### 5. Output Layer and Logits

The output layer transforms a hidden state into scores for possible output classes or tokens.

Topics include:

- Hidden-to-output weights.
- Output bias.
- Matrix multiplication.
- Output scores.
- Logits.
- Softmax activation.
- Probability distributions.
- Selecting a predicted token.

Output-layer equation:

z_t = W_hy h_t + b_y

Where:

- z_t = Output logits.
- W_hy = Hidden-to-output weight matrix.
- h_t = Current hidden state.
- b_y = Output bias.

Important distinction:

Logits are raw scores, not probabilities. Softmax can transform these scores into a probability distribution.

### 6. Softmax and Probability Distribution

Softmax converts a vector of logits into probabilities.

Formula:

softmax(z_i) = exp(z_i) / sum(exp(z_j))

Where:

- z_i = Logit for a particular output class.
- exp(z_i) = Exponential of that logit.
- The denominator = Sum of exponentials for all output classes.

The resulting probabilities are non-negative and sum to approximately 1, subject to numerical precision.

Topics include:

- Exponential calculations.
- Normalization.
- Converting logits into probabilities.
- Understanding predicted-token confidence.
- Selecting the highest-probability output.
- Numerical Softmax implementation using NumPy.

### 7. Next-Token Prediction

Next-token prediction estimates a possible next token using the preceding context.

Example:

Input:

"I love"

Possible next-token candidates:

- India
- coding
- learning
- music

The model calculates output logits and converts them into probabilities. A decoding strategy then selects a token from the predicted distribution.

Topics include:

- Input sequences.
- Context representation.
- Output logits.
- Softmax probabilities.
- Predicted tokens.
- Target tokens.
- Prediction errors.
- Training and inference.
- Sequence generation.

### 8. RNN Architectures

Different input-output arrangements are used for different sequence-processing tasks.

**One-to-One**
- One input produces one output.
- Example: Basic image classification architecture.

**One-to-Many**
- One input produces a sequence of outputs.
- Example: Generating a sequence from a single input representation.

**Many-to-One**
- A sequence of inputs produces one output.
- Example: Sentiment classification.

**Many-to-Many**
- A sequence of inputs produces multiple outputs.
- Example: Sequence labeling.

Topics include:

- Input sequences.
- Output sequences.
- Time-step alignment.
- Sequence classification.
- Sequence labeling.
- Sequence generation.

### 9. Loss Functions and Model Training

Training adjusts model parameters to reduce prediction errors.

Topics include:

- Predicted output.
- Actual target.
- Prediction error.
- Loss calculation.
- Cross-entropy loss.
- Gradients.
- Backpropagation.
- Optimizers.
- Learning rate.
- Epochs.
- Training iterations.
- Parameter updates.

A loss function measures how different the model's predictions are from the target values. During training, gradients are used by an optimizer to update the model's trainable parameters.

### 10. Backpropagation Through Time (BPTT)

Backpropagation Through Time is the method commonly used to calculate gradients through an unrolled RNN sequence.

Topics include:

- Why BPTT is required.
- Unrolling an RNN across time steps.
- Forward propagation through a sequence.
- Loss across time steps.
- Gradients flowing backward through time.
- Shared parameters across time steps.
- Gradient accumulation.
- Vanishing gradients.
- Exploding gradients.
- Gradient clipping.

BPTT allows the model to learn how its parameters affect predictions across a sequence, including the influence of earlier time steps.

### 11. Python and NumPy Practice

Practical implementations will include:

- Creating vectors and matrices.
- Array indexing.
- Matrix multiplication.
- Element-wise operations.
- Bias addition.
- Activation functions.
- Hidden-state calculations.
- Output-layer calculations.
- Softmax implementation.
- Numerical examples.
- Simple RNN implementations.
- Debugging and validating calculations.

### 12. Advanced RNN Concepts

- Limitations of Vanilla RNNs.
- Long Short-Term Memory (LSTM).
- Gated Recurrent Unit (GRU).
- Sequence modeling.
- Language modeling.
- Natural Language Processing applications.
- Time-series forecasting.
- RNN vs LSTM vs GRU.

## 🛠️ Technologies Used

- **Python** – Programming and model implementation.
- **NumPy** – Numerical calculations and matrix operations.
- **Jupyter Notebook** – Interactive coding and experiments.
- **Matplotlib** – Visualization of numerical results.
- **Pandas** – Dataset loading and analysis.
- **TensorFlow / Keras** – Planned framework-based implementations.

## 📂 Repository Structure

```text
RNN-Deep-Learning/
│
├── README.md
│
├── notes/
│   ├── 01_rnn_basics.md
│   ├── 02_tokenization_embeddings.md
│   ├── 03_many_to_many.md
│   ├── 04_bptt.md
│   └── 05_output_layer.md
│
├── src/
│   ├── output_layer.py
│   └── simple_rnn.py
│
├── notebooks/
│   └── rnn_practice.ipynb
│
└── exercises/
    └── matrix_multiplication.py
