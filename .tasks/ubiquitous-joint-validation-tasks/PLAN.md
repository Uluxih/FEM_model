# Валидация UbiquitousJointModel2D — Task Execution Plan

## Your Mission

Проверить файл `FEM/Integration_Point_Level/UbiquitousJointModel2D.py` на наличие математических ошибок и ошибок кодирования, покрыть его тестами и исправить все найденные недочеты.

**Plan File:** `.tasks/ubiquitous-joint-validation-tasks/PLAN.md`
**Tasks Directory:** `.tasks/ubiquitous-joint-validation-tasks/`

## Execution Steps

### 1. Read This Plan
Review this file for the next incomplete task, key decisions, and information from previous agents.

### 2. Understand Your Task
Read your task file: `.tasks/ubiquitous-joint-validation-tasks/task-XX-[name].md`
- **Goal** — What you are trying to achieve
- **Key Points** — Important considerations
- **Done When** — Objective acceptance criteria

### 3. Execute the Task
- Make necessary code changes
- Ensure code compiles without errors
- Verify all Done When criteria are met

### 4. Update This Plan
- Mark the task as completed in `## Task Plan`
- Add a 1-2 sentence outcome summary in `## Shared Context`
- Document only critical decisions that affect future tasks

### 5. Await Approval (MANDATORY)
Wait for user confirmation before proceeding to the next task.

### 6. Review Task List (MANDATORY)
Analyze remaining tasks based on what you learned:
- Did you encounter unexpected complexity?
- Should any tasks be split, merged, removed, or reordered?
- Are there missing tasks?

### 7. Present Review Findings (MANDATORY)
Always present your findings — even if no changes are needed — and await user approval before proceeding.

### 8. Update Task Files (if approved)
- Modify/create task files as needed
- Update `## Task Plan` in PLAN.md accordingly

---

## Task Plan

- [ ] `.tasks/ubiquitous-joint-validation-tasks/task-01-code-analysis.md`: Анализ кода и поиск синтаксических ошибок
- [ ] `.tasks/ubiquitous-joint-validation-tasks/task-02-math-validation.md`: Проверка математической модели и алгоритмов
- [ ] `.tasks/ubiquitous-joint-validation-tasks/task-03-create-tests.md`: Создание тестов для верификации
- [ ] `.tasks/ubiquitous-joint-validation-tasks/task-04-fix-errors.md`: Исправление найденных ошибок

---

## Shared Context

### Overview
Проверка и исправление модели `UbiquitousJointModel2D.py` (2D Ubiquitous-Joint Damage-Plasticity с пластичностью матрицы и Return Mapping трещины).

### Project Context
- `FEM/Integration_Point_Level/UbiquitousJointModel2D.py` — Основной файл модели материала.

### Key Decisions
- Используется пошаговый подход: сначала статический анализ и проверка математики, затем покрытие тестами, затем внесение исправлений.

### Caveats & Problems
- Требуется внимательная проверка угловых точек (apex) в Drucker-Prager и деления на ноль при вычислении жесткостей и градиентов текучести.