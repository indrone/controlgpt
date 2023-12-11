import textstat
def get_readility_score(text):
    return textstat.flesch_reading_ease(text)

def get_readility_level(text):
    return textstat.flesch_kincaid_grade(text)


def sylable_count(text):
    return textstat.syllable_count(text)
def lexion_count(text):
    return textstat.lexicon_count(text, removepunct=True)

def sent_count(text):
    return textstat.sentence_count(text)

def char_count(text):
    return textstat.letter_count(text, ignore_spaces=True)

def get_read_time(text):
    return textstat.reading_time(text, ms_per_char=14.69)

def get_average_word_len(text):
    len_data=[len(i) for i in text.split()]
    return sum(len_data)/len(len_data)



