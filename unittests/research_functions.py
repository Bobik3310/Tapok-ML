from collections import Counter
from typing import List
import math


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x) == Counter(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное произведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_product = -1
    
    for i in range(len(x) - 1):
        product = x[i] * x[i + 1]
        # Проверяем, что хотя бы один из множителей делится на 3
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            if product > max_product:
                max_product = product
    
    return max_product


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    height = len(image)
    width = len(image[0])
    channels = len(image[0][0])
    
    result = [[0.0 for _ in range(width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            weighted_sum = 0.0
            for k in range(channels):
                weighted_sum += image[i][j][k] * weights[k]
            result[i][j] = weighted_sum
    
    return result


def rle_scalar(x: List[List[int]], y: List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    # Развернуть RLE векторы
    def expand_rle(rle_vector):
        expanded = []
        for value, count in rle_vector:
            expanded.extend([value] * count)
        return expanded
    
    x_expanded = expand_rle(x)
    y_expanded = expand_rle(y)
    
    # Проверить длины
    if len(x_expanded) != len(y_expanded):
        return -1
    
    # Вычислить скалярное произведение
    result = 0
    for i in range(len(x_expanded)):
        result += x_expanded[i] * y_expanded[i]
    
    return result

def norm(X: List[float]) -> float:
    ans = 0
    for x in X:
        ans += x ** 2
    return ans ** 0.5

def dot(X: List[float], Y: List[float]) -> float:
    ans = 0
    for i in range(len(X)):
        ans += X[i] * Y[i]
    return ans

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    ans = []
    for x in X:
        ans.append([])
        for y in Y:
            cos_dist = 1
            x_norm = norm(x)
            y_norm = norm(y)
            if x_norm > 0 and y_norm > 0:
                cos_dist = dot(x, y) / (x_norm * y_norm)

            ans[-1].append(cos_dist)
    return ans
