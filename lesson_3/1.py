a = ''


def biggest_word(word):
    if len(word) == 0:
        return 0
    l = []
    w = word[0]
    for i in range(len(word)):
        if i == len(word) - 1:
            l.append(w)
            break
        if word[i] == word[i + 1]:
            w += word[i + 1]
            i += 1
        else:
            l.append(w)
            w = word[i + 1]
    d = {key: len(key) for key in l}
    return max(d.values())


print(biggest_word(a))
