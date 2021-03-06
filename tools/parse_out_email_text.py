from nltk.stem.snowball import SnowballStemmer
import string


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = []
    if len(content) > 1:
        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text_string = content[1].translate(translator)

        stemmer = SnowballStemmer('english')
        for i in text_string.split():
            words.append(stemmer.stem(i.strip()))
    return ' '.join(words)


def main():
    ff = open(r"C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\text_learning\test_email.txt", "r")
    text = parseOutText(ff)
    print(text)


if __name__ == '__main__':
    main()
