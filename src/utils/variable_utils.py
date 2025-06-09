def get_var_name(i):
    """Получение имени переменной с нижним индексом"""
    subscripts = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    if isinstance(i, (int, float)) and 1 <= i <= len(subscripts):
        return f'x{subscripts[int(i)-1]}'
    return f'x_{i}'  # fallback для больших индексов

def parse_expression(expr_str):
    """Парсинг строки выражения в коэффициенты"""
    import re
    # Заменяем - на +- для единообразия
    expr_str = expr_str.replace('-', '+-').replace(' ', '')
    # Разбиваем по +
    terms = expr_str.split('+')
    coeffs = {}
    
    for term in terms:
        if not term:
            continue
        # Ищем x с индексом (поддержка обычных и нижних индексов)
        match = re.search(r'(-?\d*\.?\d*)x[₁₂₃₄₅₆₇₈₉\d]', term)
        if match:
            coeff = match.group(1)
            if coeff in ['', '-']:
                coeff = '-1' if coeff == '-' else '1'
            
            # Определяем индекс переменной
            subscripts = {'₁': 1, '₂': 2, '₃': 3, '₄': 4, '₅': 5, '₆': 6, '₇': 7, '₈': 8, '₉': 9}
            var_idx = None
            
            # Проверяем нижние индексы
            for subscript, idx in subscripts.items():
                if subscript in term:
                    var_idx = idx
                    break
            
            # Если нижний индекс не найден, ищем обычный индекс
            if var_idx is None:
                idx_match = re.search(r'x(\d+)', term)
                if idx_match:
                    var_idx = int(idx_match.group(1))
                else:
                    continue
            
            coeffs[var_idx] = float(coeff)
    
    return coeffs

def construct_expression(terms):
    """Построение строки выражения из термов"""
    expr = []
    for var, coef in terms.items():
        if coef == 0:
            continue
        coef = format_number(coef)  # Форматируем коэффициент
        
        # Добавляем терм с учетом знака коэффициента
        if len(expr) > 0:  # Не первый терм
            if coef == 1:
                expr.append(f"+{var}")
            elif coef == -1:
                expr.append(f"-{var}")
            elif coef > 0:
                expr.append(f"+{coef}{var}")
            else:  # coef < 0
                expr.append(f"{coef}{var}")
        else:  # Первый терм
            if coef == 1:
                expr.append(var)
            elif coef == -1:
                expr.append(f"-{var}")
            else:
                expr.append(f"{coef}{var}")
    
    return " ".join(expr) if expr else "0"

def format_number(num):
    """Форматирование числа: убираем .0 для целых чисел"""
    if isinstance(num, (int, float)):
        return int(num) if float(num).is_integer() else num
    return num 