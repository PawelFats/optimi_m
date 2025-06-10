import streamlit as st
import numpy as np
from pulp import *
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx
import math

def get_var_name(i):
    """Получение имени переменной с нижним индексом"""
    subscripts = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    if isinstance(i, (int, float)) and 1 <= i <= len(subscripts):
        return f'x{subscripts[int(i)-1]}'
    return f'x_{i}'  # fallback для больших индексов

# Инициализация состояния приложения
if 'objective_terms' not in st.session_state:
    st.session_state.objective_terms = {get_var_name(i): 0 for i in range(1, 3)}

if 'constraints' not in st.session_state:
    st.session_state.constraints = [
        {'left': {get_var_name(i): 0 for i in range(1, 3)}, 'sign': '<=', 'right': 0}
    ]

if 'max_var' not in st.session_state:
    st.session_state.max_var = 2

# Добавляем хранение сохраненных задач и текущего имени задачи
if 'saved_problems' not in st.session_state:
    st.session_state.saved_problems = {}

if 'current_problem_name' not in st.session_state:
    st.session_state.current_problem_name = ""

def update_num_vars():
    """Обновление количества переменных"""
    old_max = st.session_state.max_var
    new_max = st.session_state.num_vars
    st.session_state.max_var = new_max
    
    # Обновляем целевую функцию
    new_terms = {}
    for i in range(new_max):
        var = get_var_name(i+1)
        new_terms[var] = st.session_state.objective_terms.get(var, 0)
    st.session_state.objective_terms = new_terms
    
    # Обновляем ограничения
    for constraint in st.session_state.constraints:
        new_left = {}
        for i in range(new_max):
            var = get_var_name(i+1)
            new_left[var] = constraint['left'].get(var, 0)
        constraint['left'] = new_left

def format_number(num):
    """Форматирование числа: убираем .0 для целых чисел"""
    if isinstance(num, (int, float)):
        return int(num) if float(num).is_integer() else num
    return num

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

def on_change_objective(var):
    """Обработчик изменения коэффициентов целевой функции"""
    key = f"obj_{var}"
    st.session_state.objective_terms[var] = st.session_state[key]

def on_change_constraint_coef(c_idx, var):
    """Обработчик изменения коэффициентов ограничений"""
    key = f"constr_{c_idx}_{var}"
    st.session_state.constraints[c_idx]['left'][var] = st.session_state[key]

def on_change_constraint_right(c_idx):
    """Обработчик изменения правой части ограничений"""
    key = f"rhs_{c_idx}"
    st.session_state.constraints[c_idx]['right'] = st.session_state[key]

def on_change_constraint_sign(c_idx):
    """Обработчик изменения знака ограничений"""
    key = f"sign_{c_idx}"
    st.session_state.constraints[c_idx]['sign'] = st.session_state[key]

def add_constraint():
    st.session_state.constraints.append(
        {'left': {get_var_name(i+1): 0 for i in range(st.session_state.max_var)},
         'sign': '<=',
         'right': 0}
    )

def delete_constraint(idx):
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(idx)

def parse_expression(expr_str):
    """Парсинг строки выражения в коэффициенты"""
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

def plot_constraints(constraints_str, objective_str=None, solution=None, padding=2):
    """Построение графика ограничений и ОДР"""
    # Создаем фигуру с нужными пропорциями
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Проверяем, есть ли ограничения
    constraints = [c for c in constraints_str.split('\n') if c.strip()]
    if not constraints:
        ax.text(0.5, 0.5, 'Нет ограничений', horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return fig
    
    # Находим границы для построения графика
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    # Собираем все точки пересечения ограничений и оси
    intersections_x = []
    intersections_y = []
    
    # Создаем список всех линий ограничений
    lines = []  # (a, b, right_val, sign)
    for constraint in constraints:
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
            
        coeffs = parse_expression(left)
        right_val = float(right.strip())
        
        a = coeffs.get(1, 0)  # коэффициент при x₁
        b = coeffs.get(2, 0)  # коэффициент при x₂
        
        if b != 0 or a != 0:
            lines.append((a, b, right_val, sign))
            
            # Добавляем точки пересечения с осями
            if b != 0:
                # Пересечение с осью Y (x = 0)
                y_intersect = right_val / b
                intersections_y.append(y_intersect)
                
            if a != 0:
                # Пересечение с осью X (y = 0)
                x_intersect = right_val / a
                intersections_x.append(x_intersect)
    
    # Находим точки пересечения прямых
    for i, line1 in enumerate(lines):
        a1, b1, r1, _ = line1
        for line2 in lines[i+1:]:
            a2, b2, r2, _ = line2
            # Находим точку пересечения двух линий
            det = a1*b2 - a2*b1
            if det != 0:  # Линии не параллельны
                x = (r1*b2 - r2*b1) / det
                y = (r2*a1 - r1*a2) / det
                intersections_x.append(x)
                intersections_y.append(y)
    
    # Добавляем точку решения, если она есть
    if solution and solution['status'] == "Найдено оптимальное решение":
        x = solution['variables'].get('x₁', 0)
        y = solution['variables'].get('x₂', 0)
        intersections_x.append(x)
        intersections_y.append(y)
    
    # Определяем границы графика
    if intersections_x:
        x_min = min(0, min(intersections_x)) - 1
        x_max = max(0, max(intersections_x)) + 1
    if intersections_y:
        y_min = min(0, min(intersections_y)) - 1
        y_max = max(0, max(intersections_y)) + 1
    
    # Если границы все еще не определены или слишком узкие
    if x_min == float('inf') or x_max == float('-inf') or abs(x_max - x_min) < 2:
        x_min, x_max = -5, 5
    if y_min == float('inf') or y_max == float('-inf') or abs(y_max - y_min) < 2:
        y_min, y_max = -5, 5
    
    # Создаем сетку точек для построения
    x = np.linspace(x_min, x_max, 1000)
    
    # Настраиваем цвета для ограничений (приятные пастельные тона)
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99', '#FF99CC']
    
    # Добавляем оси координат
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, zorder=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, zorder=1)
    
    # Список для хранения элементов легенды
    legend_elements = []
    
    # Рисуем ограничения
    for idx, constraint in enumerate(constraints):
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
            
        coeffs = parse_expression(left)
        right_val = float(right.strip())
        
        a = coeffs.get(1, 0)  # коэффициент при x₁
        b = coeffs.get(2, 0)  # коэффициент при x₂
        
        color = colors[idx % len(colors)]
        label = f"${left} {sign} {format_number(right_val)}$"
        
        if b != 0:
            # Выражаем x₂
            y = (-a*x + right_val) / b
            
            # Рисуем основную линию
            line = ax.plot(x, y, color=color, label=label, linewidth=2, zorder=2)[0]
            legend_elements.append(line)
            
            # Добавляем штрихи для показа направления ОДР
            if sign != '=':
                # Вычисляем нормаль к линии (направление в сторону ОДР)
                normal_x = b / np.sqrt(a**2 + b**2)
                normal_y = -a / np.sqrt(a**2 + b**2)
                
                # Если знак >=, меняем направление нормали
                if sign == '>=':
                    normal_x = -normal_x
                    normal_y = -normal_y
                
                # Создаем штрихи вдоль линии
                num_hatch = 30
                hatch_length = (x_max - x_min) / 40
                
                for t in np.linspace(0.1, 0.9, num_hatch):
                    x_base = x_min + t * (x_max - x_min)
                    y_base = (-a*x_base + right_val) / b
                    
                    # Рисуем штрих
                    ax.plot([x_base, x_base + normal_x * hatch_length],
                           [y_base, y_base + normal_y * hatch_length],
                           color=color, alpha=0.5, linewidth=1, zorder=2)
        
        elif a != 0:
            # Вертикальная линия
            x_val = right_val / a
            line = ax.axvline(x=x_val, color=color, label=label, linewidth=2, zorder=2)
            legend_elements.append(line)
            
            # Добавляем штрихи для вертикальной линии
            if sign != '=':
                direction = -1 if ((sign == '<=' and a > 0) or (sign == '>=' and a < 0)) else 1
                num_hatch = 20
                hatch_length = (x_max - x_min) / 40
                
                for y_base in np.linspace(y_min + (y_max-y_min)*0.1,
                                        y_min + (y_max-y_min)*0.9, num_hatch):
                    ax.plot([x_val, x_val + direction * hatch_length],
                           [y_base, y_base],
                           color=color, alpha=0.5, linewidth=1, zorder=2)
    
    # Если есть решение, отмечаем точку решения
    if solution and solution['status'] == "Найдено оптимальное решение":
        x1 = solution['variables'].get('x₁', 0)
        x2 = solution['variables'].get('x₂', 0)
        solution_point = ax.plot(x1, x2, 'r*', markersize=15, label='Оптимальное решение', zorder=4)[0]
        legend_elements.append(solution_point)
        
        # Добавляем подпись к точке
        ax.annotate(f'$({format_number(x1)}, {format_number(x2)})$\n$f={format_number(solution["objective"])}$',
                   (x1, x2), xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   zorder=4)
    
    # Если есть целевая функция, рисуем её градиент
    if objective_str:
        obj_coeffs = parse_expression(objective_str)
        if len(obj_coeffs) == 2:
            # Получаем коэффициенты целевой функции
            a = obj_coeffs.get(1, 0)
            b = obj_coeffs.get(2, 0)
            
            # Рисуем вектор градиента в центре графика
            center_x = (x_max + x_min) / 2
            center_y = (y_max + y_min) / 2
            gradient_length = min(x_max - x_min, y_max - y_min) / 6
            
            # Нормализуем градиент
            norm = np.sqrt(a**2 + b**2)
            if norm > 0:
                grad_x = a / norm * gradient_length
                grad_y = b / norm * gradient_length
                
                # Рисуем стрелку градиента
                arrow = ax.arrow(center_x, center_y, grad_x, grad_y,
                               head_width=gradient_length/10, head_length=gradient_length/8,
                               fc='red', ec='red', alpha=0.6, zorder=3,
                               label='Градиент целевой функции')
                legend_elements.append(arrow)
    
    # Настройка графика
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    
    # Устанавливаем пределы осей
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Делаем одинаковый масштаб по осям
    ax.set_aspect('equal')
    
    # Добавляем легенду
    if legend_elements:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Добавляем заголовок
    ax.set_title('Область допустимых решений (ОДР)', pad=20)
    
    # Настраиваем отступы
    plt.tight_layout()
    
    return fig

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
        
def plot_branch_and_bound_tree(root_node):
    """Построение дерева решений для метода ветвей и границ"""
    G = nx.Graph()
    pos = {}
    labels = {}
    
    def add_nodes_recursive(node, x=0, y=0, level=0, pos_dict=None):
        node_id = id(node)
        G.add_node(node_id)
        pos_dict[node_id] = (x, -y)
        
        # Создаем метку узла
        if node.solution:
            if node.solution['status'] == "Найдено оптимальное решение":
                label = f"f={format_number(node.solution['objective'])}\n"
                all_integer = all(abs(round(val) - val) <= 1e-10 
                                for val in node.solution['variables'].values())
                if all_integer:
                    label += "Целое решение:\n"
                for var, val in sorted(node.solution['variables'].items()):
                    label += f"{var}={format_number(val)}\n"
            else:
                label = node.solution['status']
        else:
            label = "Нет решения"
            
        if node.bound_type and node.var_name:
            label = f"{node.var_name} {node.bound_type} {format_number(node.bound_value)}\n" + label
            
        labels[node_id] = label
        
        # Добавляем ребра и рекурсивно обрабатываем детей
        width = 2 ** (level + 1)
        for i, child in enumerate(node.children):
            child_id = id(child)
            offset = width * (-0.25 + 0.5 * i)
            add_nodes_recursive(child, x + offset, y + 1, level + 1, pos_dict)
            G.add_edge(node_id, child_id)
    
    add_nodes_recursive(root_node, pos_dict=pos)
    
    # Создаем график
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # Рисуем граф
    nx.draw(G, pos, labels=labels, with_labels=True,
            node_color='lightblue', node_size=3000,
            font_size=8, font_weight='bold',
            ax=ax, node_shape='s')  # Используем квадратные узлы
    
    ax.set_title('Дерево решений метода ветвей и границ')
    plt.axis('off')  # Отключаем оси
    
    return fig

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
            
            # Для метода с округлением округляем значения
            if "с округлением" in method:
                vars_values = {var: round(val) for var, val in vars_values.items()}
                # Пересчитываем значение целевой функции с округленными значениями
                obj_val = 0
                for i, coef in obj_coeffs.items():
                    var_name = get_var_name(i)
                    obj_val += coef * vars_values[var_name]
            else:
                obj_val = value(prob.objective)
            
            result = {
                "status": status_messages.get(LpStatus[prob.status], LpStatus[prob.status]),
                "objective": obj_val,
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

# Функции для сохранения и загрузки задач
def save_current_problem(name):
    """Сохранение текущей задачи"""
    # Создаем глубокую копию всех данных задачи
    saved_data = {
        'objective_terms': {},
        'constraints': [],
        'max_var': st.session_state.max_var
    }
    
    # Копируем целевую функцию
    for var, coef in st.session_state.objective_terms.items():
        saved_data['objective_terms'][var] = coef
    
    # Копируем ограничения
    for constraint in st.session_state.constraints:
        new_constraint = {
            'left': {},
            'sign': constraint['sign'],
            'right': constraint['right']
        }
        for var, coef in constraint['left'].items():
            new_constraint['left'][var] = coef
        saved_data['constraints'].append(new_constraint)
    
    # Сохраняем задачу
    st.session_state.saved_problems[name] = saved_data
    st.session_state.current_problem_name = name

def load_problem(name):
    """Загрузка сохраненной задачи"""
    if name in st.session_state.saved_problems:
        saved_data = st.session_state.saved_problems[name]
        
        # Загружаем целевую функцию
        new_objective = {}
        for var, coef in saved_data['objective_terms'].items():
            new_objective[var] = coef
        st.session_state.objective_terms = new_objective
        
        # Загружаем ограничения
        new_constraints = []
        for constraint in saved_data['constraints']:
            new_constraint = {
                'left': {},
                'sign': constraint['sign'],
                'right': constraint['right']
            }
            for var, coef in constraint['left'].items():
                new_constraint['left'][var] = coef
            new_constraints.append(new_constraint)
        st.session_state.constraints = new_constraints
        
        # Загружаем количество переменных
        st.session_state.max_var = saved_data['max_var']
        
        # Обновляем текущее имя задачи
        st.session_state.current_problem_name = name

# Настройка страницы
st.set_page_config(page_title="Решение задач линейного программирования", layout="wide")

# Заголовок
st.title("Решение задач линейного программирования")

# Создаем колонки для элементов управления задачами
save_col1, save_col2, save_col3 = st.columns([2, 2, 1])

with save_col1:
    # Поле для ввода имени задачи с текущим значением
    problem_name = st.text_input(
        "Имя задачи",
        value=st.session_state.current_problem_name,
        key="problem_name_input"
    )
    
with save_col2:
    # Выпадающий список сохраненных задач
    saved_problems = list(st.session_state.saved_problems.keys())
    if saved_problems:
        selected_problem = st.selectbox(
            "Загрузить сохраненную задачу",
            [""] + saved_problems,
            key="problem_selector"
        )
        if selected_problem and selected_problem != st.session_state.current_problem_name:
            load_problem(selected_problem)
            st.rerun()

with save_col3:
    # Кнопка сохранения
    if st.button("Сохранить задачу", key="save_button") and problem_name:
        save_current_problem(problem_name)
        st.success(f"Задача '{problem_name}' сохранена!")

# Добавляем разделитель
st.markdown("---")

# Выбор типа задачи
problem_type = st.selectbox(
    "Выберите тип задачи:",
    ["Линейное программирование (ЛП)", "Целочисленное линейное программирование (ЦЛП)"],
    key="problem_type"
)

# Выбор метода решения
if problem_type == "Линейное программирование (ЛП)":
    method = "Симплекс-метод"
else:
    method = st.selectbox(
        "Выберите метод решения:",
        ["Метод ветвей и границ", "Графический метод"],
        key="method"
    )

# Выбор типа оптимизации
optimization_type = st.selectbox(
    "Тип оптимизации:",
    ["Максимум", "Минимум"],
    key="optimization_type"
)

# Ввод количества переменных
st.number_input("Количество переменных:", 
                min_value=2, 
                value=st.session_state.max_var,
                key="num_vars",
                on_change=update_num_vars)

# Ввод целевой функции
st.subheader("Целевая функция")
st.markdown("Введите коэффициенты целевой функции:")

# Отображение текущего вида целевой функции
objective_expr = construct_expression(st.session_state.objective_terms)
st.latex(f"f(x) = {objective_expr}")

# Ввод коэффициентов целевой функции
obj_cols = st.columns(4)
for i, var in enumerate(sorted(st.session_state.objective_terms.keys())):
    with obj_cols[i % 4]:
        key = f"obj_{var}"
        st.number_input(
            f"Коэффициент при {var}",
            value=st.session_state.objective_terms[var],
            key=key,
            on_change=on_change_objective,
            args=(var,)
        )

# Ввод ограничений
st.markdown("""
<style>
    .constraint-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .constraint-header {
        font-weight: bold;
        color: #0e1117;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.subheader("Ограничения")

if st.button("Добавить ограничение"):
    add_constraint()
    st.rerun()

# Отображение ограничений
for c_idx, constraint in enumerate(st.session_state.constraints):
    with st.container():
        st.markdown(f'<div class="constraint-container">', unsafe_allow_html=True)
        st.markdown(f'<p class="constraint-header">Ограничение {c_idx + 1}</p>', unsafe_allow_html=True)
        
        # Отображение текущего вида ограничения
        constr_expr = construct_expression(constraint['left'])
        st.latex(f"{constr_expr} {constraint['sign']} {format_number(constraint['right'])}")
        
        cols = st.columns([3, 1, 1])
        
        # Кнопки управления ограничением
        with cols[1]:
            sign_key = f"sign_{c_idx}"
            st.selectbox(
                "Знак",
                options=['<=', '=', '>='],
                index=['<=', '=', '>='].index(constraint['sign']),
                key=sign_key,
                on_change=on_change_constraint_sign,
                args=(c_idx,)
            )
        with cols[2]:
            if st.button("Удалить", key=f"del_constr_{c_idx}"):
                delete_constraint(c_idx)
                st.rerun()
        
        # Ввод коэффициентов ограничения
        term_cols = st.columns(4)
        for i, (var, coef) in enumerate(sorted(constraint['left'].items())):
            with term_cols[i % 4]:
                coef_key = f"constr_{c_idx}_{var}"
                st.number_input(
                    f"Коэффициент при {var}",
                    value=coef,
                    key=coef_key,
                    on_change=on_change_constraint_coef,
                    args=(c_idx, var)
                )
        
        # Правая часть ограничения
        rhs_key = f"rhs_{c_idx}"
        st.number_input(
            "Правая часть",
            value=constraint['right'],
            key=rhs_key,
            on_change=on_change_constraint_right,
            args=(c_idx,)
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Добавляем отступ между ограничениями

# Преобразование данных в строковый формат для решателя
objective = construct_expression({var: coef for var, coef in st.session_state.objective_terms.items()})
constraints_str = "\n".join(
    f"{construct_expression(c['left'])} {c['sign']} {format_number(c['right'])}"
    for c in st.session_state.constraints
)

# Кнопка решения
if st.button("Решить"):
    try:
        # Отображаем задачу в математическом виде
        st.subheader("Поставленная задача:")
        latex_max_min = "\\max" if optimization_type == "Максимум" else "\\min"
        st.latex(latex_max_min + " f(x) = " + objective)
        st.latex("\\text{при ограничениях:}")
        for constraint in st.session_state.constraints:
            constr_expr = construct_expression(constraint['left'])
            st.latex(f"{constr_expr} {constraint['sign']} {format_number(constraint['right'])}")
        if problem_type == "Целочисленное линейное программирование (ЦЛП)":
            st.latex("x_1, x_2 \\in \\mathbb{Z}")
        
        if problem_type == "Линейное программирование (ЛП)":
            result = solve_lp_standard(objective, constraints_str, optimization_type)
            
            # Вывод результатов
            st.subheader("Результаты:")
            st.write(f"Статус решения: {result['status']}")
            
            if result['status'] == "Найдено оптимальное решение":
                st.latex(f"f(x^*) = {format_number(result['objective'])}")
                st.write("Значения переменных:")
                for var, val in sorted(result['variables'].items()):
                    st.latex(f"{var} = {format_number(val)}")
            
            # Построение графика для задач с двумя переменными
            if st.session_state.max_var == 2:
                st.subheader("Графическое представление:")
                fig = plot_constraints(constraints_str, objective, result)
                st.pyplot(fig)
        
        else:  # Целочисленное линейное программирование (ЦЛП)
            if method == "Метод ветвей и границ":
                result, root = solve_branch_and_bound(objective, constraints_str, optimization_type)
                
                # Вывод результатов
                st.subheader("Результаты:")
                st.write(f"Статус решения: {result['status']}")
                
                if result['status'] == "Найдено оптимальное решение":
                    st.latex(f"f(x^*) = {format_number(result['objective'])}")
                    st.write("Значения переменных:")
                    for var, val in sorted(result['variables'].items()):
                        st.latex(f"{var} = {format_number(val)}")
                
                # Построение дерева решений
                st.subheader("Дерево решений:")
                fig = plot_branch_and_bound_tree(root)
                st.pyplot(fig)
            
            else:  # Графический метод для ЦЛП
                # Построение графика для задач с двумя переменными
                if st.session_state.max_var == 2:
                    # Находим границы области для поиска целых точек
                    x_min, x_max, y_min, y_max = -10, 10, -10, 10  # Базовые границы
                    
                    # Создаем функцию для проверки точки на удовлетворение всем ограничениям
                    def check_point(x, y):
                        for constraint in st.session_state.constraints:
                            left_expr = sum(coef * (x if var == 'x₁' else y) 
                                          for var, coef in constraint['left'].items())
                            right_val = constraint['right']
                            
                            if constraint['sign'] == '<=' and left_expr > right_val:
                                return False
                            elif constraint['sign'] == '>=' and left_expr < right_val:
                                return False
                            elif constraint['sign'] == '=' and left_expr != right_val:
                                return False
                        return True
                    
                    # Находим все целые точки в ОДР
                    integer_points = []
                    for x in range(int(x_min), int(x_max) + 1):
                        for y in range(int(y_min), int(y_max) + 1):
                            if check_point(x, y):
                                # Вычисляем значение целевой функции
                                obj_val = sum(coef * (x if var == 'x₁' else y) 
                                            for var, coef in st.session_state.objective_terms.items())
                                integer_points.append((x, y, obj_val))
                    
                    if integer_points:
                        # Сортируем точки по значению целевой функции
                        integer_points.sort(key=lambda p: p[2], 
                                         reverse=(optimization_type == "Максимум"))
                        
                        # Формируем результат для отображения на графике
                        best_point = integer_points[0]
                        result = {
                            "status": "Найдено оптимальное решение",
                            "objective": best_point[2],
                            "variables": {"x₁": best_point[0], "x₂": best_point[1]}
                        }
                        
                        # Выводим оптимальное решение
                        st.subheader("Результаты:")
                        st.write("Статус решения: Найдено оптимальное решение")
                        st.latex(f"f(x^*) = {format_number(best_point[2])}")
                        st.write("Значения переменных:")
                        st.latex(f"x_1^* = {best_point[0]}")
                        st.latex(f"x_2^* = {best_point[1]}")
                        
                        # Построение графика
                        st.subheader("Графическое представление:")
                        fig = plot_constraints(constraints_str, objective, result)
                        st.pyplot(fig)
                        
                        # Выводим все целочисленные точки
                        st.subheader("Точки с целочисленными координатами в ОДР:")
                        st.write("Проверяем точки с целочисленными координатами, удовлетворяющие всем ограничениям:")
                        for x, y, obj_val in integer_points:
                            st.latex(f"x_1 = {x}, x_2 = {y}: f(x) = {format_number(obj_val)}")
                    else:
                        st.write("В области допустимых решений нет точек с целочисленными координатами.")
                else:
                    st.error("Графический метод доступен только для задач с двумя переменными.")
            
    except Exception as e:
        st.error(f"Произошла ошибка при решении задачи: {str(e)}")
        st.error("Пожалуйста, проверьте правильность ввода данных")
        
        # Даже при ошибке пытаемся построить график для задач с двумя переменными
        if st.session_state.max_var == 2:
            try:
                st.subheader("Графическое представление:")
                fig = plot_constraints(constraints_str, objective, None)
                st.pyplot(fig)
            except Exception as plot_error:
                st.error(f"Не удалось построить график: {str(plot_error)}") 