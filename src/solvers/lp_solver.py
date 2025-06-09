from pulp import *
from src.utils.variable_utils import parse_expression, get_var_name

def solve_lp_standard(objective_str, constraints_str, optimization_type):
    """Решение задачи линейного программирования стандартным методом"""
    try:
        # Создаем задачу
        prob = LpProblem("LP_Problem", LpMaximize if optimization_type == "Максимум" else LpMinimize)
        
        # Парсим целевую функцию
        obj_coeffs = parse_expression(objective_str)
        if not obj_coeffs:
            return {
                "status": "Ошибка: целевая функция пуста",
                "objective": None,
                "variables": None
            }
        
        # Создаем переменные
        max_var = max(obj_coeffs.keys())
        vars_dict = LpVariable.dicts("x", range(1, max_var + 1))
        
        # Добавляем целевую функцию
        prob += lpSum([obj_coeffs[i] * vars_dict[i] for i in obj_coeffs])
        
        # Добавляем базовые ограничения
        constraints_added = False
        for constraint in constraints_str.split('\n'):
            if not constraint.strip():
                continue
                
            # Разделяем левую и правую части
            if '<=' in constraint:
                left, right = constraint.split('<=')
                is_less_equal = True
                is_equal = False
            elif '>=' in constraint:
                left, right = constraint.split('>=')
                is_less_equal = False
                is_equal = False
            elif '=' in constraint:
                left, right = constraint.split('=')
                is_equal = True
            else:
                continue
                
            left_coeffs = parse_expression(left)
            if not left_coeffs:
                continue
                
            try:
                right_val = float(right.strip())
            except ValueError:
                continue
            
            # Добавляем ограничение
            expr = lpSum([left_coeffs[i] * vars_dict[i] for i in left_coeffs])
            if is_equal:
                prob += expr == right_val
            elif is_less_equal:
                prob += expr <= right_val
            else:
                prob += expr >= right_val
            constraints_added = True
        
        # Всегда добавляем ограничения неотрицательности
        for i in range(1, max_var + 1):
            prob += vars_dict[i] >= 0
            constraints_added = True
        
        if not constraints_added:
            return {
                "status": "Ошибка: нет корректных ограничений",
                "objective": None,
                "variables": None
            }
        
        # Решаем задачу
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Формируем результат
        status_messages = {
            "Optimal": "Найдено оптимальное решение",
            "Not Solved": "Задача не решена",
            "Infeasible": "Задача не имеет допустимых решений (система ограничений несовместна)",
            "Unbounded": "Задача не имеет решения (целевая функция не ограничена)",
            "Undefined": "Решение не определено"
        }
        
        if LpStatus[prob.status] == 'Optimal':
            # Получаем значения переменных
            vars_values = {}
            for i in range(1, max_var + 1):
                var_name = get_var_name(i)
                vars_values[var_name] = value(vars_dict[i])
            
            result = {
                "status": status_messages.get(LpStatus[prob.status], LpStatus[prob.status]),
                "objective": value(prob.objective),
                "variables": vars_values
            }
        else:
            result = {
                "status": status_messages.get(LpStatus[prob.status], LpStatus[prob.status]),
                "objective": None,
                "variables": None
            }
        
        return result
        
    except Exception as e:
        return {
            "status": f"Ошибка при решении: {str(e)}",
            "objective": None,
            "variables": None
        } 