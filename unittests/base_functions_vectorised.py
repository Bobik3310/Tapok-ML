import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    # Выбираем каждый 4-й элемент по оси 0 (строкам)
    rows_selected = X[::4, :]
    
    # Выбираем элементы с 120 по 500 с шагом 5 по оси 1 (столбцам)
    cols_selected = rows_selected[:, 120:500:5]
    
    return cols_selected

def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    # Получаем главную диагональ
    diagonal = np.diag(X)
    
    # Выбираем неотрицательные элементы
    non_neg_elements = diagonal[diagonal >= 0]
    
    # Если есть неотрицательные элементы, возвращаем их сумму, иначе -1
    if len(non_neg_elements) > 0:
        return int(np.sum(non_neg_elements))
    else:
        return -1


def replace_values(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    # Создаем копию массива
    result = X.copy()
    
    # Вычисляем средние по столбцам (axis=0)
    column_means = np.mean(X, axis=0)
    
    # Вычисляем границы для каждого столбца
    lower_bounds = 0.25 * column_means
    upper_bounds = 1.5 * column_means
    
    # Создаем маску для замены
    mask = (result < lower_bounds) | (result > upper_bounds)
    
    # Заменяем значения по маске
    result[mask] = -1.0
    
    return result
