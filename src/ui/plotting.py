import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from src.utils.variable_utils import parse_expression, format_number

def plot_constraints(constraints_str, objective_str=None, solution=None, padding=2):
    """Построение графика ограничений и ОДР"""
    # Создаем фигуру с нужными пропорциями
    fig = plt.figure(figsize=(12, 8))
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
    
    # Собираем все точки пересечения ограничений
    intersections_x = []
    intersections_y = []
    
    # Создаем список всех линий ограничений для поиска пересечений
    lines = []
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
        
        if b != 0:
            # y = (-ax + r)/b
            lines.append((a, b, right_val, sign))
        elif a != 0:
            # Вертикальная линия x = r/a
            x_val = right_val / a
            x_min = min(x_min, x_val)
            x_max = max(x_max, x_val)
    
    # Находим точки пересечения
    for i, line1 in enumerate(lines):
        a1, b1, r1, _ = line1
        # Пересечение с осями
        if b1 != 0:
            # x = 0: y = r1/b1
            intersections_y.append(r1/b1)
            # y = 0: x = r1/a1
            if a1 != 0:
                intersections_x.append(r1/a1)
        
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
    
    # Определяем границы графика на основе найденных точек
    if intersections_x:
        x_min = min(min(intersections_x), x_min)
        x_max = max(max(intersections_x), x_max)
    if intersections_y:
        y_min = min(min(intersections_y), y_min)
        y_max = max(max(intersections_y), y_max)
    
    # Если границы все еще не определены или слишком узкие
    if x_min == float('inf') or x_max == float('-inf') or x_min == x_max:
        x_min, x_max = -10, 10
    if y_min == float('inf') or y_max == float('-inf') or y_min == y_max:
        y_min, y_max = -10, 10
    
    # Добавляем отступы
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding/10
    x_max += x_range * padding/10
    y_min -= y_range * padding/10
    y_max += y_range * padding/10
    
    # Создаем сетку точек
    x = np.linspace(x_min, x_max, 1000)
    
    # Цвета для разных ограничений (пастельные тона)
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF', '#99FFFF']
    
    # Добавляем оси координат
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, zorder=1)
    
    # Список для хранения элементов легенды
    legend_elements = []
    
    # Рисуем ограничения и их градиенты
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
        
        # Получаем коэффициенты
        a = coeffs.get(1, 0)  # коэффициент при x₁
        b = coeffs.get(2, 0)  # коэффициент при x₂
        
        # Нормализуем градиент
        norm = np.sqrt(a**2 + b**2)
        if norm > 0:
            grad_x = a / norm
            grad_y = b / norm
        else:
            continue
            
        color = colors[idx % len(colors)]
        label = f"${left} {sign} {format_number(right_val)}$"
            
        if b != 0:
            # Выражаем x₂
            y = (-a*x + right_val) / b
            # Рисуем линию
            line = ax.plot(x, y, color=color, label=label, linewidth=2, zorder=2)[0]
            legend_elements.append(line)
            
            # Создаем "расчёску" вдоль линии
            if sign != '=':
                # Вычисляем нормаль к линии
                normal_x = -b / norm
                normal_y = a / norm
                
                # Создаем точки вдоль линии для "зубцов"
                num_teeth = 40  # количество зубцов
                teeth_length = (x_max - x_min) / 30  # длина зубцов
                
                for t in np.linspace(0.1, 0.9, num_teeth):
                    x_base = x_min + t * (x_max - x_min)
                    y_base = (-a*x_base + right_val) / b
                    
                    # Определяем направление зубцов в зависимости от знака неравенства
                    direction = 1 if sign == '<=' else -1
                    
                    # Рисуем зубец
                    ax.plot([x_base, x_base + direction * normal_x * teeth_length],
                           [y_base, y_base + direction * normal_y * teeth_length],
                           color=color, alpha=0.3, linewidth=1, zorder=2)
                
        elif a != 0:
            # Вертикальная линия
            x_val = right_val / a
            # Рисуем линию
            line = ax.axvline(x=x_val, color=color, label=label, linewidth=2, zorder=2)
            legend_elements.append(line)
            
            # Создаем "расчёску" для вертикальной линии
            if sign != '=':
                direction = 1 if sign == '<=' else -1
                num_teeth = 20
                teeth_length = (x_max - x_min) / 30
                
                for y_base in np.linspace(y_min + (y_max-y_min)*0.1, 
                                        y_min + (y_max-y_min)*0.9, num_teeth):
                    ax.plot([x_val, x_val + direction * teeth_length],
                           [y_base, y_base],
                           color=color, alpha=0.3, linewidth=1, zorder=2)
    
    # Если есть решение, отмечаем точку решения
    if solution and solution['status'] == "Найдено оптимальное решение":
        x1 = solution['variables'].get('x₁', 0)
        x2 = solution['variables'].get('x₂', 0)
        solution_point = ax.plot(x1, x2, 'r*', markersize=15, label='Оптимальное решение', zorder=3)[0]
        legend_elements.append(solution_point)
        
        # Добавляем подпись к точке
        ax.annotate(f'$({format_number(x1)}, {format_number(x2)})$\n$f={format_number(solution["objective"])}$',
                   (x1, x2), xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   zorder=3)
    
    # Если есть целевая функция, рисуем её линии уровня
    if objective_str:
        obj_coeffs = parse_expression(objective_str)
        if len(obj_coeffs) == 2:
            a = obj_coeffs.get(1, 0)
            b = obj_coeffs.get(2, 0)
            if b != 0:
                # Вычисляем хорошие значения для линий уровня
                if solution and solution['status'] == "Найдено оптимальное решение":
                    optimal_value = solution['objective']
                    levels = np.linspace(optimal_value - abs(optimal_value), 
                                       optimal_value + abs(optimal_value), 5)
                else:
                    levels = np.linspace(-10, 10, 5)
                
                for level in levels:
                    y = (-a*x + level) / b
                    ax.plot(x, y, 'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # Настройка графика
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlabel('$x_{1}$')
    ax.set_ylabel('$x_{2}$')
    
    # Устанавливаем пределы осей
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Делаем одинаковый масштаб по осям
    ax.set_aspect('equal')
    
    # Добавляем легенду только если есть элементы для нее
    if legend_elements:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Добавляем заголовок
    ax.set_title('Область допустимых решений (ОДР)', pad=20)
    
    # Настраиваем отступы
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