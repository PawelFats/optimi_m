import streamlit as st
from src.utils.variable_utils import get_var_name

def init_session_state():
    """Инициализация состояния приложения"""
    if 'objective_terms' not in st.session_state:
        st.session_state.objective_terms = {get_var_name(i): 0 for i in range(1, 3)}

    if 'constraints' not in st.session_state:
        st.session_state.constraints = [
            {'left': {get_var_name(i): 0 for i in range(1, 3)}, 'sign': '<=', 'right': 0}
        ]

    if 'max_var' not in st.session_state:
        st.session_state.max_var = 2

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