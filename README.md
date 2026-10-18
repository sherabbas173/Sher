Mushroom Classification using CNN and RNN

Course project from DigiSkills "Artificial Intelligence using Python" (Batch-04, Hands-on Exercise No. 04).

What it does
- Loads and explores the mushroom dataset (8124 samples, 22 categorical features, target: edible or poisonous)
- Preprocesses the data with label encoding, a 70/30 train/test split, and reshaping to `(samples, 22, 1)` for CNN and RNN input
- Trains a CNN (Adam, batch size 30, 10 epochs) and an RNN (RMSprop, batch size 40, 20 epochs) using TensorFlow/Keras
- Evaluates both models with accuracy, confusion matrices and classification reports
- Compares the models with matplotlib charts (test accuracy, training accuracy and loss curves)
- Tokenizes text from a JSON file using a Hugging Face Transformers tokenizer (`bert-base-uncased`)

Results

| Model | Batch size | Epochs | Optimizer | Test accuracy |
|-------|-----------|--------|-----------|---------------|
| CNN   | 30        | 10     | Adam      | 100% (1.0000) |
| RNN   | 40        | 20     | RMSprop   | 99.92% (0.9992) |

- The CNN classified all 2438 test samples correctly.
- The RNN made 2 mistakes, both poisonous mushrooms predicted as edible.
- The CNN reached its result with fewer epochs, so it is the best model for this dataset.

Text tokenization
The second part loads `exercise_text.json` and tokenizes its text with the bert-base-uncased tokenizer. The text produced 173 tokens, and the notebook shows the tokens, token IDs, attention mask and the decoded text.

How to run
1. Open `mushroom_classification_cnn_rnn.ipynb` in Google Colab using the "Open in Colab" button.
2. Download `mushrooms.csv` and `exercise_text.json` from the course LMS and upload them to Colab. The data files are not included in this repository.
3. Run all cells in order.

Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, matplotlib, Hugging Face Transformers
