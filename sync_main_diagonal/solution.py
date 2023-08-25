import re
import time
import numpy as np
import sklearn.metrics.pairwise
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer


module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
print("Loading multilingual model...")
model = hub.load(module_url)

L_POS = 0
R_POS = 1
L_WORDS = 2
R_WORDS = 3
L_a = 4
L_b = 5


def embed_text(input):
    return model(input)


def count_of_eps(inp, a, b):
    delta_range = range(-6, 7)
    delta = [(a + delta_a, b + delta_b) for delta_a in delta_range for delta_b in delta_range]
    result = 0
    for d in delta:
        try:
            if inp[d[0]][d[1]] == 255:
                result += 1
        except IndexError:
            pass
    return result


def filtered_main_diag(filename):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    img = np.asarray(Image.open(filename))
    res = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i, j] == 255:
                res[i, j] = count_of_eps(img, i, j)
    res_min_max = min_max_scaler.fit_transform(res)
    for i in range(len(res_min_max)):
        for j in range(len(res_min_max[i])):
            if res_min_max[i, j] < 1.0:
                res_min_max[i, j] = 0
    return res_min_max


def find_max_path(alist, a=0, b=0, path=None):
    if path is None:
        path = []
    while (a < len(alist)) and (b < len(alist[a])):
        if alist[a][b] == 100:
            path.append({"a": a, "b": b})
            b += 1
        else:
            try:
                if alist[a + 1][b + 1] == 100:
                    a += 1
                    b += 1
                    continue
            except IndexError:
                pass
            diag_a = a + 1
            diag_b = b + 1
            diag_i = 1
            while (diag_a + diag_i < len(alist)) and (diag_b + diag_i < len(alist[diag_a])):
                if alist[diag_a + diag_i][diag_b + diag_i] == 100:
                    diag_a = diag_a + diag_i
                    diag_b = diag_b + diag_i
                    diag_i += 1
                    break
                diag_i += 1
            check = []
            for j in range(b + 1, len(alist[a])):
                if alist[a][j] == 100:
                    check.append({"a": a, "b": j})
                    break
            if not check:
                a += 1
                continue
            i = a + 1
            while i < len(alist):
                if alist[i][b] == 100:
                    check.append({"a": i, "b": b})
                    break
                i += 1
            path_result = []
            for c in check:
                path_new = path[:]
                path_new.append({"a": c["a"], "b": c["b"]})
                path_result.append(find_max_path(alist, c["a"] + 1, c["b"] + 1, path_new))
            if diag_i != 1:
                path_result.append(find_max_path(alist, diag_a, diag_b, path))
            maximum = 0
            max_path = []
            for p in path_result:
                if len(p) > maximum:
                    maximum = len(p)
                    max_path = p[:]
            if max_path:
                return max_path
        a += 1
    return path


def distance(mx, my, a, b):
    return abs(b*my + a*mx)/((a*a + b*b)**0.5)


def find_max_path_v2(array):
    buffer = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 100:
                buffer.append((distance(j, i, -len(array), len(array[i])), j, i))
                array[i][j] = 0
    buffer.sort()
    path = []
    for x in buffer[0:20*int((len(array)**2 + len(array[0])**2)**0.5)]:
        path.append((x[1], x[2]))
    path.sort()
    result = []
    for p in path:
        result.append({"a": p[1], "b": p[0]})
        array[p[1]][p[0]] = 1
    return array


def create_sync_v2(synchronize, L_word, R_word, L_end, R_end, L_len, R_len, L_window=50):
    sync1 = []
    L = 0
    R = 0
    R_window = int(L_window * (len(R_word) / len(L_word)))
    maxtime = {"max_time": 0, "L": L, "R": R}
    while (L < len(L_word)) and (R < len(R_word)):
        t1 = time.perf_counter()
        p = find_max_path(synchronize[L:L + L_window, R:R + R_window])
        t2 = time.perf_counter()
        if t2 - t1 > maxtime["max_time"]:
            maxtime = {"max_time": t2 - t1, "L": L, "R": R}
        a = -1
        b = -1
        for i in p:
            a = i["a"]
            b = i["b"]
            sync1.append([L_end[L + a],
                          R_end[R + b],
                          L_word[L + a],
                          R_word[R + b],
                          L + a,
                          R + b])
            print(sync1[-1][L_POS], sync1[-1][R_POS], sep="::")
            print(sync1[-1][L_WORDS])
            print(sync1[-1][R_WORDS])
        if (a == -1) or (b == -1):
            break
        L = L + a + 1
        R = R + b + 1
        print(f"L={L}, R={R}")
        print(maxtime)
    print(f"len(L_word)={len(L_word)}")
    print(f"len(R_word)={len(R_word)}")
    print(f"len(sync)={len(sync1)}")

    return sync1


def get_sim_v2():
    with open("rus.orig.html", mode="r", encoding="UTF-8") as f:
        rus_orig = f.read()
    with open("eng.orig.html", mode="r", encoding="UTF-8") as f:
        eng_orig = f.read()

    labels_1 = re.findall(r"<p>(.*?)</p>", rus_orig, flags=re.DOTALL | re.UNICODE)
    labels_2 = re.findall(r"<p>(.*?)</p>", eng_orig, flags=re.DOTALL | re.UNICODE)
    embeddings_1 = embed_text(labels_1)
    embeddings_2 = embed_text(labels_2)

    sim = 1 - np.arccos(
        sklearn.metrics.pairwise.cosine_similarity(embeddings_1,
                                                   embeddings_2)) / np.pi
    for i in range(len(sim)):
        for j in range(len(sim[i])):
            if sim[i][j] < 0.63:
                sim[i][j] = 0
            else:
                sim[i][j] = 100
    L_end = []
    length = 0
    for i in labels_1:
        length += len(i)
        L_end.append(length)
    R_end = []
    length = 0
    for i in labels_2:
        length += len(i)
        R_end.append(length)

    return sim, labels_1, labels_2, L_end, R_end


def main():
    synchronize, L_word, R_word, L_end, R_end = get_sim_v2()
    synchronize = find_max_path_v2(synchronize)
    img = Image.fromarray(np.uint8(synchronize * 255), 'L')
    img.save(f"input.png")

    image = filtered_main_diag("input.png")
    synchronize = np.asarray(np.uint8(image * 100))
    image = Image.fromarray(np.uint8(image * 255))
    image.save("output1.png")

    two_sync = create_sync_v2(synchronize, L_word, R_word, L_end, R_end,
                              len(L_word) - 1, len(R_word) - 1)
    for i in range(len(synchronize)):
        for j in range(len(synchronize[i])):
            synchronize[i][j] = 0
    for i in two_sync:
        synchronize[i[L_a]][i[L_b]] = 255
    img = Image.fromarray(np.uint8(synchronize), 'L')
    img.save("output2.png")


if __name__ == "__main__":
    main()
