from typing import List
from copy import deepcopy

def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    result = []
    
    # Выбираем каждый 4-й элемент по оси n (строкам)
    for i in range(0, len(X), 4):
        row = []
        # Выбираем элементы с 120 по 500 с шагом 5 по оси m (столбцам)
        for j in range(120, min(500, len(X[i])), 5):
            row.append(X[i][j])
        result.append(row)
    
    return result

def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    n = len(X)
    m = len(X[0]) if n > 0 else 0
    min_dim = min(n, m)
    
    diagonal_sum = 0
    found_non_neg = False
    
    for i in range(min_dim):
        element = X[i][i]
        if element >= 0:
            diagonal_sum += element
            found_non_neg = True
    
    return diagonal_sum if found_non_neg else -1


def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    if not X or not X[0]:
        return deepcopy(X)
    
    n = len(X)
    m = len(X[0])
    
    # Создаем глубокую копию
    result = deepcopy(X)
    
    # Вычисляем средние по столбцам
    column_means = []
    for j in range(m):
        column_sum = 0.0
        for i in range(n):
            column_sum += X[i][j]
        column_means.append(column_sum / n)
    
    # Заменяем значения по условию
    for j in range(m):
        M = column_means[j]
        lower_bound = 0.25 * M
        upper_bound = 1.5 * M
        
        for i in range(n):
            value = result[i][j]
            if value < lower_bound or value > upper_bound:
                result[i][j] = -1.0
    
    return result
