import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from src.utils.variable_utils import parse_expression, format_number

import numpy as np
import matplotlib.pyplot as plt

# Обновленная функция для красивого отображения ограничений с рисками, указывающими область допустимых решений
def plot_constraints(constraints_str, objective_str=None, solution=None, padding=2):
    """Построение графика ограничений и ОДР с рисками вдоль каждой границы"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Парсим ограничения
    constraints = [c.strip() for c in constraints_str.split('\n') if c.strip()]
    if not constraints:
        ax.text(0.5, 0.5, 'Нет ограничений', ha='center', va='center')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return fig

    # Собираем коэффициенты линий и точки пересечений для определения границ
    lines = []  # (a, b, r, sign)
    intersections = []
    for c in constraints:
        if '<=' in c:
            lhs, rhs = c.split('<='); sign = '<='
        elif '>=' in c:
            lhs, rhs = c.split('>='); sign = '>='
        elif '=' in c:
            lhs, rhs = c.split('='); sign = '='
        else:
            continue
        coeffs = parse_expression(lhs)
        r = float(rhs)
        a = coeffs.get(1, 0)
        b = coeffs.get(2, 0)
        if a==0 and b==0:
            continue
        lines.append((a, b, r, sign))
        # Пересечения с осями
        if b != 0:
            intersections.append((0, r/b))
        if a != 0:
            intersections.append((r/a, 0))

    # Считаем межлинией
    for i in range(len(lines)):
        a1, b1, r1, _ = lines[i]
        for j in range(i+1, len(lines)):
            a2, b2, r2, _ = lines[j]
            det = a1*b2 - a2*b1
            if abs(det) > 1e-6:
                x = (r1*b2 - r2*b1) / det
                y = (a1*r2 - a2*r1) / det
                intersections.append((x, y))

    # Добавляем решение
    if solution and solution.get('status') == 'Найдено оптимальное решение':
        x_opt = solution['variables'].get('x₁', 0)
        y_opt = solution['variables'].get('x₂', 0)
        intersections.append((x_opt, y_opt))

    # Вычисляем границы области
    xs, ys = zip(*intersections) if intersections else ([-10,10],[-10,10])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max: x_min -= 5; x_max += 5
    if y_min == y_max: y_min -= 5; y_max += 5
    dx, dy = x_max - x_min, y_max - y_min
    x_min -= dx*padding/10; x_max += dx*padding/10
    y_min -= dy*padding/10; y_max += dy*padding/10

    x_vals = np.linspace(x_min, x_max, 1000)
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99']

    # Оси
    ax.axhline(0, color='black', lw=1, alpha=0.3)
    ax.axvline(0, color='black', lw=1, alpha=0.3)

    # Рисуем ограничения
    for idx, (a, b, r, sign) in enumerate(lines):
        color = colors[idx % len(colors)]
        label = f"{a}x₁ + {b}x₂ {sign} {r}"

        # Функция линии
        if b != 0:
            y_line = (r - a*x_vals) / b
            ax.plot(x_vals, y_line, color=color, lw=2, label=label)
        else:
            x_line = r / a
            ax.axvline(x_line, color=color, lw=2, label=label)

        # Рисуем риски вдоль линии:
        # Градиент нормали направления = (a, b) / ||(a,b)||
        norm = np.hypot(a, b)
        if norm == 0: continue
        grad_x, grad_y = a/norm, b/norm
        # Направление для рисок: для '<=' риски в сторону, противоположную градиенту
        dir_mul = -1 if sign == '<=' else (1 if sign == '>=' else 0)
        tick_dx, tick_dy = grad_x * dir_mul, grad_y * dir_mul

        # Параметры рисок
        num_ticks = 20
        tick_len = min(dx, dy) / 30
        t_vals = np.linspace(0.1, 0.9, num_ticks)

        for t in t_vals:
            # Точка на линии
            if b != 0:
                x0 = x_min + t*(x_max - x_min)
                y0 = (r - a*x0)/b
            else:
                x0 = r/a
                y0 = y_min + t*(y_max - y_min)
            x1 = x0 + tick_dx * tick_len
            y1 = y0 + tick_dy * tick_len
            ax.plot([x0, x1], [y0, y1], color=color, lw=1, alpha=0.7)

    # Помечаем решение
    if solution and solution.get('status') == 'Найдено оптимальное решение':
        ax.plot(x_opt, y_opt, 'r*', ms=15, label='Оптимум')
        ax.annotate(f'({x_opt:.2f}, {y_opt:.2f})', (x_opt, y_opt), textcoords='offset points', xytext=(10,10))

    # Уровни целевой функции (без изменений)
    if objective_str:
        obj = parse_expression(objective_str)
        if len(obj)==2:
            a0, b0 = obj.get(1,0), obj.get(2,0)
            levels = np.linspace(-10, 10, 5)
            for lvl in levels:
                if b0!=0:
                    ax.plot(x_vals, (lvl - a0*x_vals)/b0, 'k--', alpha=0.3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_title('Область допустимых решений (c рисками)')
    plt.tight_layout()
    return fig


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