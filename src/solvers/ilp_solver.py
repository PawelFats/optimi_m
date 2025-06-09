from pulp import *
import math
from src.utils.variable_utils import parse_expression, get_var_name

class BranchAndBoundNode:
    def __init__(self, constraints, parent=None, bound_type=None, bound_value=None, var_name=None):
        self.constraints = constraints
        self.parent = parent
        self.children = []
        self.solution = None
        self.bound_type = bound_type  # '<=' или '>='
        self.bound_value = bound_value
        self.var_name = var_name
        
    def add_child(self, child):
        self.children.append(child)

def solve_branch_and_bound(objective_str, constraints_str, optimization_type):
    """Решение задачи методом ветвей и границ"""
    try:
        # Создаем задачу для корневого узла
        prob = LpProblem("LP_Problem", LpMaximize if optimization_type == "Максимум" else LpMinimize)
        
        # Парсим целевую функцию
        obj_coeffs = parse_expression(objective_str)
        if not obj_coeffs:
            return {
                "status": "Ошибка: целевая функция пуста",
                "objective": None,
                "variables": None
            }, None
        
        # Создаем переменные
        max_var = max(obj_coeffs.keys())
        vars_dict = LpVariable.dicts("x", range(1, max_var + 1))
        
        # Добавляем целевую функцию
        prob += lpSum([obj_coeffs[i] * vars_dict[i] for i in obj_coeffs])
        
        # Добавляем базовые ограничения
        for constraint in constraints_str.split('\n'):
            if not constraint.strip():
                continue
            
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
            
            expr = lpSum([left_coeffs[i] * vars_dict[i] for i in left_coeffs])
            if is_equal:
                prob += expr == right_val
            elif is_less_equal:
                prob += expr <= right_val
            else:
                prob += expr >= right_val
        
        # Добавляем ограничения неотрицательности
        for i in range(1, max_var + 1):
            prob += vars_dict[i] >= 0
        
        root = BranchAndBoundNode(constraints_str)
        root.solution = {"status": "Не решено"}
        
        def solve_node(node, current_prob, depth=0):
            if depth > 20:  # Ограничение глубины
                return
            
            # Решаем задачу для текущего узла
            try:
                current_prob.solve(PULP_CBC_CMD(msg=False))
                
                if LpStatus[current_prob.status] == 'Optimal':
                    node.solution = {
                        "status": "Найдено оптимальное решение",
                        "objective": value(current_prob.objective),
                        "variables": {f"x₁": value(vars_dict[1]), f"x₂": value(vars_dict[2])}
                    }
                    
                    # Проверяем, есть ли нецелые переменные
                    non_integer_vars = {}
                    for i, var in vars_dict.items():
                        val = value(var)
                        if val is not None and abs(round(val) - val) > 1e-10:
                            non_integer_vars[f"x₁" if i == 1 else f"x₂"] = val
                    
                    if not non_integer_vars:  # Если все переменные целые
                        return
                    
                    # Выбираем переменную с наибольшей дробной частью
                    var_to_branch = max(non_integer_vars.items(), 
                                      key=lambda x: abs(round(x[1]) - x[1]))[0]
                    value_to_branch = non_integer_vars[var_to_branch]
                    
                    # Создаем два новых узла
                    floor_val = math.floor(value_to_branch)
                    ceil_val = math.ceil(value_to_branch)
                    
                    # Левая ветвь (x ≤ floor)
                    left_prob = current_prob.copy()
                    var_idx = 1 if var_to_branch == "x₁" else 2
                    left_prob += vars_dict[var_idx] <= floor_val
                    left_node = BranchAndBoundNode(node.constraints + f"\n{var_to_branch} <= {floor_val}",
                                                 node, '<=', floor_val, var_to_branch)
                    node.add_child(left_node)
                    solve_node(left_node, left_prob, depth + 1)
                    
                    # Правая ветвь (x ≥ ceil)
                    right_prob = current_prob.copy()
                    right_prob += vars_dict[var_idx] >= ceil_val
                    right_node = BranchAndBoundNode(node.constraints + f"\n{var_to_branch} >= {ceil_val}",
                                                  node, '>=', ceil_val, var_to_branch)
                    node.add_child(right_node)
                    solve_node(right_node, right_prob, depth + 1)
                else:
                    node.solution = {
                        "status": "Решение не найдено",
                        "objective": None,
                        "variables": None
                    }
            except Exception as e:
                node.solution = {
                    "status": f"Ошибка: {str(e)}",
                    "objective": None,
                    "variables": None
                }
        
        # Решаем начальную задачу
        solve_node(root, prob)
        
        # Находим лучшее целочисленное решение
        best_solution = None
        best_objective = float('-inf') if optimization_type == "Максимум" else float('inf')
        
        def find_best_solution(node):
            nonlocal best_solution, best_objective
            
            if node.solution and node.solution["status"] == "Найдено оптимальное решение":
                # Проверяем, что все переменные целые
                all_integer = all(abs(round(val) - val) <= 1e-10 
                                for val in node.solution["variables"].values())
                
                if all_integer:
                    if optimization_type == "Максимум":
                        if node.solution["objective"] > best_objective:
                            best_objective = node.solution["objective"]
                            best_solution = node.solution
                    else:
                        if node.solution["objective"] < best_objective:
                            best_objective = node.solution["objective"]
                            best_solution = node.solution
            
            for child in node.children:
                find_best_solution(child)
        
        find_best_solution(root)
        
        if best_solution is None:
            return {
                "status": "Целочисленное решение не найдено",
                "objective": None,
                "variables": None
            }, root
        
        return best_solution, root
        
    except Exception as e:
        return {
            "status": f"Ошибка при решении: {str(e)}",
            "objective": None,
            "variables": None
        }, None

def solve_graphical_method(objective_str, constraints_str, optimization_type):
    """Решение задачи целочисленного программирования графическим методом"""
    try:
        # Находим границы области для поиска целых точек
        x_min, x_max, y_min, y_max = -10, 10, -10, 10  # Базовые границы
        
        # Создаем функцию для проверки точки на удовлетворение всем ограничениям
        def check_point(x, y):
            for constraint in constraints_str.split('\n'):
                if not constraint.strip():
                    continue
                    
                if '<=' in constraint:
                    left, right = constraint.split('<=')
                    sign = '<='
                elif '>=' in constraint:
                    left, right = constraint.split('>=')
                    sign = '>='
                elif '=' in constraint:
                    left, right = constraint.split('=')
                    sign = '='
                else:
                    continue
                
                left_coeffs = parse_expression(left)
                if not left_coeffs:
                    continue
                
                try:
                    right_val = float(right.strip())
                except ValueError:
                    continue
                
                left_expr = sum(coef * (x if i == 1 else y) for i, coef in left_coeffs.items())
                
                if sign == '<=' and left_expr > right_val:
                    return False
                elif sign == '>=' and left_expr < right_val:
                    return False
                elif sign == '=' and left_expr != right_val:
                    return False
            return True
        
        # Находим все целые точки в ОДР
        integer_points = []
        obj_coeffs = parse_expression(objective_str)
        
        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                if check_point(x, y):
                    # Вычисляем значение целевой функции
                    obj_val = sum(coef * (x if i == 1 else y) for i, coef in obj_coeffs.items())
                    integer_points.append((x, y, obj_val))
        
        if not integer_points:
            return {
                "status": "В области допустимых решений нет точек с целочисленными координатами",
                "objective": None,
                "variables": None
            }
        
        # Сортируем точки по значению целевой функции
        integer_points.sort(key=lambda p: p[2], reverse=(optimization_type == "Максимум"))
        
        # Формируем результат
        best_point = integer_points[0]
        result = {
            "status": "Найдено оптимальное решение",
            "objective": best_point[2],
            "variables": {"x₁": best_point[0], "x₂": best_point[1]},
            "all_points": integer_points
        }
        
        return result
        
    except Exception as e:
        return {
            "status": f"Ошибка при решении: {str(e)}",
            "objective": None,
            "variables": None
        } 