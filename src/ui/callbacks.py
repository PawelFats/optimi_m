import streamlit as st

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
    """Добавление нового ограничения"""
    st.session_state.constraints.append(
        {'left': {f"x₁": 0, f"x₂": 0},
         'sign': '<=',
         'right': 0}
    )

def delete_constraint(idx):
    """Удаление ограничения"""
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(idx) 