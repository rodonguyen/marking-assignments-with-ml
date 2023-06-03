from tensorflow.keras.preprocessing.text import Tokenizer
import re

def clean_punctuation(x):
    copy = x.copy()
    for number, t in enumerate(copy):
        t = re.sub(r"[^a-z0-9'\":,\.?!]+", " ", str(t).lower())
        t = re.sub(r"\*", " * ", t)
        t = re.sub(r"\'", " ' ", t)
        t = re.sub(r"\"", " \" ", t)
        t = re.sub(r"\:", " : ", t)        
        t = re.sub(r"\,", " , ", t)
        t = re.sub(r"\.", " . ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\!", " ! ", t)
        copy.iloc[number] = t
    return copy


def tokenise(x_train, x_val, x_test, char_level=False):
    x_train_las = clean_punctuation(x_train)
    x_val_las = clean_punctuation(x_val)
    x_test_las = clean_punctuation(x_test)

    tokenizer = Tokenizer(num_words=50000,
                          filters='$&()+/<=>[\\]^_`{|}~\t',
                          char_level=char_level)
    tokenizer.fit_on_texts(x_train_las)

    train_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_train_las):
        train_sequences.append(seq)

    val_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_val_las):
        val_sequences.append(seq)

    test_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_test_las):
        test_sequences.append(seq)

    max_length = max(find_max_list(train_sequences),
                     find_max_list(val_sequences),
                     find_max_list(test_sequences))

    x_train_tokenised = np.array(pad_sequences(
        train_sequences, maxlen=max_length, padding='post'))
    x_val_tokenised = np.array(pad_sequences(
        val_sequences, maxlen=max_length, padding='post'))
    x_test_tokenised = np.array(pad_sequences(
        test_sequences, maxlen=max_length, padding='post'))

    return x_train_tokenised, x_val_tokenised, x_test_tokenised, \
        x_train_las, x_val_las, x_test_las, max_length, tokenizer
