import streamlit as st
from src.models.problem import init_session_state, update_num_vars, save_current_problem, load_problem
from src.utils.variable_utils import construct_expression, format_number
from src.solvers.lp_solver import solve_lp_standard
from src.solvers.ilp_solver import solve_branch_and_bound, solve_graphical_method
from src.ui.plotting import plot_constraints, plot_branch_and_bound_tree
from src.ui.callbacks import (
    on_change_objective, on_change_constraint_coef,
    on_change_constraint_right, on_change_constraint_sign,
    add_constraint, delete_constraint
)

def main():
    # Настройка страницы
    st.set_page_config(page_title="Решение задач ЛП и ЦЛП", layout="wide")

    # Заголовок
    st.title("Решение задач ЛП и ЦЛП")

    # Инициализация состояния приложения
    init_session_state()

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
            latex_max_min = "\\max" if optimization_type == 'Максимум' else "\\min"
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
                        result = solve_graphical_method(objective, constraints_str, optimization_type)
                        
                        if result['status'] == "Найдено оптимальное решение":
                            # Выводим оптимальное решение
                            st.subheader("Результаты:")
                            st.write("Статус решения: Найдено оптимальное решение")
                            st.latex(f"f(x^*) = {format_number(result['objective'])}")
                            st.write("Значения переменных:")
                            st.latex(f"x_1^* = {format_number(result['variables']['x₁'])}")
                            st.latex(f"x_2^* = {format_number(result['variables']['x₂'])}")
                            
                            # Построение графика
                            st.subheader("Графическое представление:")
                            fig = plot_constraints(constraints_str, objective, result)
                            st.pyplot(fig)
                            
                            # Выводим все целочисленные точки
                            st.subheader("Точки с целочисленными координатами в ОДР:")
                            st.write("Проверяем точки с целочисленными координатами, удовлетворяющие всем ограничениям:")
                            for x, y, obj_val in result['all_points']:
                                st.latex(f"x_1 = {x}, x_2 = {y}: f(x) = {format_number(obj_val)}")
                        else:
                            st.write(result['status'])
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

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: grey;'>© 2025 Автор проекта: Павел Фатьянов, Новосибирский государственный технический университет</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: grey;'>Содействие оказала Казанская Ольга Васильевна.</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 