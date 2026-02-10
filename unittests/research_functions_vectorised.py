import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное произведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    # Вычислить произведения соседних элементов
    products = x[:-1] * x[1:]
    
    # Найти маску, где хотя бы один из соседних элементов делится на 3
    mask = (x[:-1] % 3 == 0) | (x[1:] % 3 == 0)
    
    # Применить маску к произведениям
    valid_products = products[mask]
    
    # Если нет подходящих произведений, вернуть -1
    if len(valid_products) == 0:
        return -1
    
    # Вернуть максимальное произведение
    return np.max(valid_products)


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    # Умножить каждый канал на соответствующий вес и просуммировать по каналам
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    # Развернуть RLE векторы
    def expand_rle(rle_vector):
        values = []
        counts = []
        for pair in rle_vector:
            values.append(pair[0])
            counts.append(pair[1])
        
        values = np.array(values)
        counts = np.array(counts)
        
        # Создать развернутый вектор
        expanded = np.repeat(values, counts)
        return expanded
    
    x_expanded = expand_rle(x)
    y_expanded = expand_rle(y)
    
    # Проверить длины
    if len(x_expanded) != len(y_expanded):
        return -1
    
    # Вычислить скалярное произведение
    return np.dot(x_expanded, y_expanded)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]

    x_norm = np.reshape(np.linalg.norm(X, axis=1), (n, 1))
    y_norm = np.reshape(np.linalg.norm(Y, axis=1), (1, m))

    x_norm = np.repeat(x_norm, m, axis=1)
    y_norm = np.repeat(y_norm, n, axis=0)

    ans = X @ Y.T
    mask = (x_norm == 0) + (y_norm == 0)
    x_norm[mask] = 1
    y_norm[mask] = 1
    ans[mask] = 1
    return ans / (x_norm * y_norm)
