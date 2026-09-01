# Алгоритмически согласованная касательная матрица модели `UbiquitousJointModel2D`

## 1. Назначение документа

Этот документ выводит алгоритмически согласованную касательную матрицу для дискретного конститутивного алгоритма, реализованного в файле
`FEM/Integration_Point_Level/UbiquitousJointModel2D.py`.

Модель последовательно применяет три нелинейных отображения:

1. return mapping пластичности матрицы по критерию Друкера—Прагера;
2. многоповерхностный return mapping шва в его локальных координатах;
3. преобразование эффективных напряжений шва в номинальные посредством анизотропной повреждённости.

Алгоритмически согласованный оператор — это производная именно этого дискретного отображения при фиксированном подтверждённом состоянии:

$$
\mathbf D_{\mathrm{alg}}^g
=
\frac{\partial \boldsymbol\sigma_{\mathrm{nom}}^g}
     {\partial \boldsymbol\varepsilon^g}.
$$

В пределах одной гладкой ветви он обеспечивает локальную линеаризацию

$$
\delta\boldsymbol\sigma_{\mathrm{nom}}^g
=
\mathbf D_{\mathrm{alg}}^g\,
\delta\boldsymbol\varepsilon^g
+o\!\left(\lVert\delta\boldsymbol\varepsilon^g\rVert\right).
$$

Производная строится при фиксированных:

- committed-переменных на начале шага;
- ориентации активного шва;
- активном наборе поверхностей;
- знаках нормального и касательного напряжений;
- ветвях функций `abs`, `min`, `max`, saturation, irreversibility и caps.

На границах переключения отображение лишь кусочно-гладкое. Там классическая производная может отсутствовать, и код использует односторонний Jacobian принятой ветви.

---

## 2. Соглашения Voigt

### 2.1 Глобальные векторы

В двумерной постановке плоского напряжённого состояния используются векторы

$$
\boldsymbol\sigma^g
=
\begin{bmatrix}
\sigma_{xx} & \sigma_{yy} & \tau_{xy}
\end{bmatrix}^{\mathsf T},
\qquad
\boldsymbol\varepsilon^g
=
\begin{bmatrix}
\varepsilon_{xx} & \varepsilon_{yy} & \gamma_{xy}
\end{bmatrix}^{\mathsf T}.
$$

Третья компонента деформации является инженерной деформацией сдвига,
$\gamma_{xy}=2\varepsilon_{xy}$, тогда как третья компонента напряжения равна физическому касательному напряжению $\tau_{xy}$.

Упругая матрица интактного материала имеет вид

$$
\mathbf D_r
=
\begin{bmatrix}
\dfrac{E}{1-\nu^2} & \dfrac{E\nu}{1-\nu^2} & 0 \\
\dfrac{E\nu}{1-\nu^2} & \dfrac{E}{1-\nu^2} & 0 \\
0 & 0 & \dfrac{E}{2(1+\nu)}
\end{bmatrix}.
$$

В коде эта матрица хранится как `D_rock`.

### 2.2 Локальная система шва

Пусть единичная нормаль к шву имеет компоненты $(n_x,n_y)$. Локальный вектор напряжений записывается как

$$
\boldsymbol\sigma^l
=
\begin{bmatrix}
\sigma_n & \sigma_p & \tau
\end{bmatrix}^{\mathsf T},
$$

где $\sigma_n$ — нормальное напряжение поперёк шва, $\sigma_p$ — нормальное напряжение вдоль шва, $\tau$ — касательное напряжение на шве.

Код строит матрицу преобразования напряжений

$$
\mathbf T_\sigma
=
\begin{bmatrix}
 n_x^2 & n_y^2 & 2n_xn_y \\
 n_y^2 & n_x^2 & -2n_xn_y \\
 -n_xn_y & n_xn_y & n_x^2-n_y^2
\end{bmatrix}.
$$

Тогда

$$
\boldsymbol\sigma^l=\mathbf T_\sigma\boldsymbol\sigma^g.
$$

Для инженерного Voigt-вектора деформаций вводятся

$$
\mathbf N=\operatorname{diag}(1,1,2),
\qquad
\mathbf N^{-1}=\operatorname{diag}\!\left(1,1,\frac12\right),
$$

$$
\mathbf T_\varepsilon
=
\mathbf N\mathbf T_\sigma\mathbf N^{-1}.
$$

В реализации выполняется энергетическая сопряжённость

$$
\mathbf T_\varepsilon=\mathbf T_\sigma^{-\mathsf T},
$$

поэтому

$$
\boldsymbol\varepsilon^l
=
\mathbf T_\varepsilon\boldsymbol\varepsilon^g,
\qquad
\boldsymbol\sigma^g
=
\mathbf T_\varepsilon^{\mathsf T}\boldsymbol\sigma^l.
$$

Равенство виртуальных работ сохраняется:

$$
(\boldsymbol\sigma^g)^{\mathsf T}\delta\boldsymbol\varepsilon^g
=
(\boldsymbol\sigma^l)^{\mathsf T}\delta\boldsymbol\varepsilon^l.
$$

Локальная упругая матрица определяется кодом как

$$
\mathbf C
=
\mathbf D_{\mathrm{local}}
=
\mathbf T_\sigma\mathbf D_r\mathbf T_\varepsilon^{-1}.
$$

Используемые в return mapping шва модули равны

$$
E_n=C_{00},
\qquad
G_s=C_{22}.
$$

Для изотропной матрицы поворот не создаёт нормального-сдвигового взаимодействия, поэтому $C_{02}=C_{20}=C_{12}=C_{21}=0$. Однако универсальные формулы ниже записаны через полную матрицу $\mathbf C$.

---

## 3. Цепная структура дискретного алгоритма

На одном trial-вызове `update_state` код выполняет цепочку

$$
\boldsymbol\varepsilon^g
\longmapsto
\boldsymbol\sigma_{\mathrm{tr}}^g
\longmapsto
\boldsymbol\sigma_m^g
\longmapsto
\boldsymbol\sigma_{\mathrm{tr}}^l
\longmapsto
\widetilde{\boldsymbol\sigma}^{l}
\longmapsto
\boldsymbol\sigma_{\mathrm{nom}}^{l}
\longmapsto
\boldsymbol\sigma_{\mathrm{nom}}^{g}.
$$

Пробное эффективное напряжение матрицы:

$$
\boldsymbol\sigma_{\mathrm{tr}}^g
=
\boldsymbol\sigma_{\mathrm{eff,old}}^g
+
\mathbf D_r\Delta\boldsymbol\varepsilon^g.
$$

Обозначим Jacobian return mapping матрицы

$$
\mathbf A_m
=
\frac{\partial\boldsymbol\sigma_m^g}
     {\partial\boldsymbol\sigma_{\mathrm{tr}}^g}.
$$

После проекции на шов

$$
\boldsymbol\sigma_{\mathrm{tr}}^l
=
\mathbf T_\sigma\boldsymbol\sigma_m^g.
$$

Полный локальный Jacobian шва, включая повреждённость, обозначим

$$
\mathbf J_J
=
\frac{\partial\boldsymbol\sigma_{\mathrm{nom}}^l}
     {\partial\boldsymbol\sigma_{\mathrm{tr}}^l}.
$$

Поскольку

$$
\boldsymbol\sigma_{\mathrm{nom}}^g
=
\mathbf T_\varepsilon^{\mathsf T}
\boldsymbol\sigma_{\mathrm{nom}}^l,
$$

цепное правило даёт основную формулу:

$$
\boxed{
\mathbf D_{\mathrm{alg}}^g
=
\mathbf T_\varepsilon^{\mathsf T}
\mathbf J_J
\mathbf T_\sigma
\mathbf A_m
\mathbf D_r
}.
$$

Именно это произведение собирается в текущем `update_state`:

```python
self.D_tangent = (
    self.T_eps.T @ joint_jacobian @ self.T_sig
    @ matrix_jacobian @ self.D_rock
)
```

Если шов не активирован, остаётся только вклад матрицы:

$$
\boxed{
\mathbf D_{\mathrm{alg}}^g
=
\mathbf A_m\mathbf D_r
}.
$$

---

## 4. Return mapping матрицы Друкера—Прагера

### 4.1 Инвариантные переменные

Код вводит

$$
p
=
\frac{\sigma_{xx}+\sigma_{yy}}{2},
\qquad
s
=
\frac{\sigma_{xx}-\sigma_{yy}}{2},
\qquad
q
=
\sqrt{s^2+\tau_{xy}^2}.
$$

Определим

$$
\mathbf y
=
\begin{bmatrix}
p & s & \tau_{xy}
\end{bmatrix}^{\mathsf T}
=
\mathbf A\boldsymbol\sigma,
$$

$$
\mathbf A
=
\begin{bmatrix}
1/2 & 1/2 & 0 \\
1/2 & -1/2 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Обратное отображение задаётся матрицей

$$
\boldsymbol\sigma
=
\mathbf B\mathbf y,
\qquad
\mathbf B
=
\begin{bmatrix}
1 & 1 & 0 \\
1 & -1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Для плоской упругой матрицы

$$
K_{2D}
=
\frac{D_{r,00}+D_{r,01}}{2},
\qquad
G=D_{r,22}.
$$

Критерий текучести матрицы:

$$
F_m
=
q+p\sin\phi_m-c_m\cos\phi_m.
$$

### 4.2 Упругая ветвь

При

$$
F_m^{\mathrm{tr}}\le 10^{-10}
$$

код оставляет пробное напряжение неизменным:

$$
\boldsymbol\sigma_m^g
=
\boldsymbol\sigma_{\mathrm{tr}}^g,
\qquad
\boxed{\mathbf A_m=\mathbf I}.
$$

Соответственно,

$$
\mathbf D_m^{\mathrm{alg}}=\mathbf D_r.
$$

### 4.3 Гладкий возврат на коническую поверхность

Введём сокращения

$$
a_m=\sin\phi_m,
\qquad
b_m=\sin\psi_m,
$$

$$
d_m
=
G+K_{2D}a_mb_m.
$$

Пластический множитель текущего дискретного алгоритма равен

$$
\Delta\lambda_m
=
\frac{F_m^{\mathrm{tr}}}{d_m}.
$$

Исправленные инварианты:

$$
p'
=
p-K_{2D}b_m\Delta\lambda_m,
$$

$$
q'
=
q-G\Delta\lambda_m.
$$

При $q>0$ направление в плоскости $(s,\tau)$ сохраняется. Обозначим

$$
\mathbf u
=
\begin{bmatrix}
u_s & u_\tau
\end{bmatrix}^{\mathsf T}
=
\frac{1}{q}
\begin{bmatrix}
s & \tau
\end{bmatrix}^{\mathsf T},
$$

$$
r=\frac{q'}{q}.
$$

Тогда

$$
s'=rs,
\qquad
\tau'=r\tau.
$$

Градиент пластического множителя по $\mathbf y=[p,s,\tau]^{\mathsf T}$:

$$
\mathbf g_\lambda^{\mathsf T}
=
\frac{\partial\Delta\lambda_m}{\partial\mathbf y}
=
\frac{1}{d_m}
\begin{bmatrix}
a_m & u_s & u_\tau
\end{bmatrix}.
$$

Для производной радиального коэффициента используем

$$
r=1-\frac{G\Delta\lambda_m}{q}.
$$

Отсюда

$$
\mathbf g_r^{\mathsf T}
=
\frac{\partial r}{\partial\mathbf y}
=
\frac{G\Delta\lambda_m}{q^2}
\begin{bmatrix}
0 & u_s & u_\tau
\end{bmatrix}
-
\frac{G}{q}\mathbf g_\lambda^{\mathsf T}.
$$

Полный Jacobian в пространстве инвариантов равен

$$
\mathbf J_y
=
\frac{\partial[p',s',\tau']^{\mathsf T}}
     {\partial[p,s,\tau]^{\mathsf T}}.
$$

Его строки имеют вид

$$
(\mathbf J_y)_{0:}
=
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix}
-
K_{2D}b_m\mathbf g_\lambda^{\mathsf T},
$$

$$
(\mathbf J_y)_{1:}
=
r
\begin{bmatrix}
0 & 1 & 0
\end{bmatrix}
+s\mathbf g_r^{\mathsf T},
$$

$$
(\mathbf J_y)_{2:}
=
r
\begin{bmatrix}
0 & 0 & 1
\end{bmatrix}
+\tau\mathbf g_r^{\mathsf T}.
$$

Следовательно, Jacobian отображения пробного напряжения в исправленное напряжение равен

$$
\boxed{
\mathbf A_m
=
\mathbf B\mathbf J_y\mathbf A
}.
$$

А касательная матрицы относительно глобальной деформации равна

$$
\boxed{
\mathbf D_m^{\mathrm{alg}}
=
\mathbf A_m\mathbf D_r
=
\mathbf B\mathbf J_y\mathbf A\mathbf D_r
}.
$$

Члены $s\mathbf g_r^{\mathsf T}$ и $\tau\mathbf g_r^{\mathsf T}$ принципиальны: они учитывают изменение направления девиаторной части пробного напряжения. Их исключение даёт лишь приближённый, а не согласованный Jacobian.

### 4.4 Возврат в вершину `apex`

Код сначала вычисляет гладкий прогноз $(p',q')$, а затем выбирает apex-ветвь, если

$$
q'<0
\qquad\text{или}\qquad
p'>p_{\mathrm{apex}}.
$$

При $\sin\phi_m>0$

$$
p_{\mathrm{apex}}
=
\frac{c_m\cos\phi_m}{\sin\phi_m}.
$$

Если $\sin\phi_m$ практически равен нулю, код принимает $p_{\mathrm{apex}}=+\infty$.

На apex-ветви

$$
q'=0,
\qquad
p'=\min(p_{\mathrm{apex}},p).
$$

Внутри фиксированной ветви производная в инвариантном пространстве равна

$$
\mathbf J_y^{\mathrm{apex}}
=
\operatorname{diag}(\alpha,0,0),
$$

где

$$
\alpha
=
\begin{cases}
1, & p<p_{\mathrm{apex}},\\
0, & p>p_{\mathrm{apex}}.
\end{cases}
$$

Поэтому

$$
\boxed{
\mathbf A_m^{\mathrm{apex}}
=
\mathbf B\mathbf J_y^{\mathrm{apex}}\mathbf A
}.
$$

После умножения на $\mathbf D_r$ получается

$$
\boxed{
\mathbf D_m^{\mathrm{apex}}
=
\alpha K_{2D}
\begin{bmatrix}
1 & 1 & 0 \\
1 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
}.
$$

На границе $p=p_{\mathrm{apex}}$, при $q=0$ и на границе переключения smooth/apex обычная производная не единственна.

---

## 5. Многоповерхностный return mapping шва

### 5.1 Поверхности и направления течения

Далее $\boldsymbol\sigma^{\mathrm{tr}}$ означает локальное пробное напряжение шва, а $\widetilde{\boldsymbol\sigma}$ — локальное эффективное напряжение после возврата:

$$
\boldsymbol\sigma^{\mathrm{tr}}
=
\begin{bmatrix}
\sigma_n^{\mathrm{tr}} & \sigma_p^{\mathrm{tr}} & \tau^{\mathrm{tr}}
\end{bmatrix}^{\mathsf T}.
$$

Введём

$$
a=\tan\phi,
\qquad
b=\tan\psi,
\qquad
r_\tau=\operatorname{sign}(\tau^{\mathrm{tr}}).
$$

В коде при $\tau^{\mathrm{tr}}=0$ принимается $r_\tau=+1$.

Текущие пределы в начале шага:

$$
f_t^{\mathrm{cur}}=f_t+q_{\mathrm{old}},
$$

$$
c^{\mathrm{cur}}=c(q_{\mathrm{old}}).
$$

Важно: `c_curr` вычисляется именно по $q_{\mathrm{old}}$, а не по $q_{\mathrm{new}}$. Поэтому в Jacobian фактической реализации отсутствует производная $dc/dq_{\mathrm{new}}$.

Пробные значения поверхностей:

$$
F_t^{\mathrm{tr}}
=
\sigma_n^{\mathrm{tr}}-f_t-q_{\mathrm{old}},
$$

$$
F_c^{\mathrm{tr}}
=
-\sigma_n^{\mathrm{tr}}-f_c,
$$

$$
F_s^{\mathrm{tr}}
=
|\tau^{\mathrm{tr}}|+a\sigma_n^{\mathrm{tr}}-c(q_{\mathrm{old}}).
$$

Нормали поверхностей по напряжениям:

$$
\mathbf n_t
=
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix}^{\mathsf T},
\qquad
\mathbf n_c
=
\begin{bmatrix}
-1 & 0 & 0
\end{bmatrix}^{\mathsf T},
$$

$$
\mathbf n_s
=
\begin{bmatrix}
a & 0 & r_\tau
\end{bmatrix}^{\mathsf T}.
$$

Направления пластического течения:

$$
\mathbf m_t
=
\begin{bmatrix}
1 & 0 & 0
\end{bmatrix}^{\mathsf T},
\qquad
\mathbf m_c
=
\begin{bmatrix}
-1 & 0 & 0
\end{bmatrix}^{\mathsf T},
$$

$$
\mathbf m_s
=
\begin{bmatrix}
b & 0 & r_\tau
\end{bmatrix}^{\mathsf T}.
$$

Для растяжения и сжатия течение ассоциированное, а для сдвига в общем случае неассоциированное, поскольку $a\ne b$.

Обновление упрочнения:

$$
q_{\mathrm{new}}
=
q_{\mathrm{old}}+H_t\Delta\lambda_t.
$$

### 5.2 Универсальный active-set вывод

Пусть $\mathcal A$ — фактически принятый набор активных механизмов. Соберём матрицы

$$
\mathbf M_{\mathcal A}
=
\begin{bmatrix}
\mathbf m_{i_1} & \cdots & \mathbf m_{i_r}
\end{bmatrix},
$$

$$
\mathbf N_{\mathcal A}
=
\begin{bmatrix}
\mathbf n_{i_1} & \cdots & \mathbf n_{i_r}
\end{bmatrix}.
$$

Эффективное напряжение после пластической коррекции:

$$
\widetilde{\boldsymbol\sigma}
=
\boldsymbol\sigma^{\mathrm{tr}}
-
\mathbf C\mathbf M_{\mathcal A}
\Delta\boldsymbol\lambda_{\mathcal A}.
$$

Линеаризация условий консистентности даёт

$$
\mathbf H_{\mathcal A}
\,d\Delta\boldsymbol\lambda_{\mathcal A}
=
\mathbf N_{\mathcal A}^{\mathsf T}
\,d\boldsymbol\sigma^{\mathrm{tr}},
$$

где

$$
\mathbf H_{\mathcal A}
=
\mathbf N_{\mathcal A}^{\mathsf T}
\mathbf C
\mathbf M_{\mathcal A}
+
\mathbf H_q.
$$

В фактической реализации $\mathbf H_q$ содержит только добавку $H_t$ в позиции, соответствующей уравнению растяжения и множителю $\Delta\lambda_t$.

Производная активных множителей по пробному локальному напряжению:

$$
\boxed{
\mathbf L_{\mathcal A}
=
\frac{\partial\Delta\boldsymbol\lambda_{\mathcal A}}
     {\partial\boldsymbol\sigma^{\mathrm{tr}}}
=
\mathbf H_{\mathcal A}^{-1}
\mathbf N_{\mathcal A}^{\mathsf T}
}.
$$

Если $\mathbf H_{\mathcal A}$ вырождена, production-код использует псевдообратную матрицу.

Jacobian эффективного return mapping шва:

$$
\boxed{
\mathbf P_J
=
\frac{\partial\widetilde{\boldsymbol\sigma}}
     {\partial\boldsymbol\sigma^{\mathrm{tr}}}
=
\mathbf I
-
\mathbf C\mathbf M_{\mathcal A}
\mathbf H_{\mathcal A}^{-1}
\mathbf N_{\mathcal A}^{\mathsf T}
}.
$$

Если требуется производная по локальной деформации, то

$$
\boxed{
\mathbf K_{ep}^{J}
=
\mathbf P_J\mathbf C
=
\mathbf C
-
\mathbf C\mathbf M_{\mathcal A}
\mathbf H_{\mathcal A}^{-1}
\mathbf N_{\mathcal A}^{\mathsf T}
\mathbf C
}.
$$

Код `_joint_jacobian` определяет активный набор по условию

$$
\Delta\lambda_i>10^{-14}.
$$

Строки полной матрицы производных множителей, соответствующие неактивным механизмам, заполняются нулями.

### 5.3 Упругий набор

Если все три поверхности допустимы,

$$
F_t^{\mathrm{tr}}\le\mathrm{tol},
\qquad
F_c^{\mathrm{tr}}\le\mathrm{tol},
\qquad
F_s^{\mathrm{tr}}\le\mathrm{tol},
$$

то

$$
\Delta\boldsymbol\lambda=\mathbf0,
\qquad
\widetilde{\boldsymbol\sigma}=\boldsymbol\sigma^{\mathrm{tr}},
$$

$$
\boxed{\mathbf P_J=\mathbf I}.
$$

### 5.4 Чистое растяжение `t`

Для $\mathcal A=\{t\}$

$$
\mathbf H_t
=
\mathbf n_t^{\mathsf T}\mathbf C\mathbf m_t+H_t
=
E_n+H_t.
$$

Пластический множитель:

$$
\Delta\lambda_t
=
\frac{F_t^{\mathrm{tr}}}{E_n+H_t}.
$$

Jacobian:

$$
\boxed{
\mathbf P_J^t
=
\mathbf I
-
\frac{\mathbf C\mathbf m_t\mathbf n_t^{\mathsf T}}
     {E_n+H_t}
}.
$$

### 5.5 Чистое сжатие `c`

Для $\mathcal A=\{c\}$

$$
\mathbf H_c
=
\mathbf n_c^{\mathsf T}\mathbf C\mathbf m_c
=
E_n.
$$

$$
\Delta\lambda_c
=
\frac{F_c^{\mathrm{tr}}}{E_n},
$$

$$
\boxed{
\mathbf P_J^c
=
\mathbf I
-
\frac{\mathbf C\mathbf m_c\mathbf n_c^{\mathsf T}}
     {E_n}
}.
$$

### 5.6 Чистый сдвиг `s`

Для $\mathcal A=\{s\}$

$$
\mathbf H_s
=
\mathbf n_s^{\mathsf T}\mathbf C\mathbf m_s.
$$

Для структуры $\mathbf C$, используемой моделью,

$$
\mathbf H_s
=
G_s+E_nab.
$$

$$
\Delta\lambda_s
=
\frac{F_s^{\mathrm{tr}}}{G_s+E_nab},
$$

$$
\boxed{
\mathbf P_J^s
=
\mathbf I
-
\frac{\mathbf C\mathbf m_s\mathbf n_s^{\mathsf T}}
     {G_s+E_nab}
}.
$$

При $a\ne b$ эта матрица в общем случае несимметрична.

### 5.7 Угол растяжение–сдвиг `t+s`

Для $\mathcal A=\{t,s\}$

$$
\mathbf M_{ts}
=
\begin{bmatrix}
\mathbf m_t & \mathbf m_s
\end{bmatrix},
\qquad
\mathbf N_{ts}
=
\begin{bmatrix}
\mathbf n_t & \mathbf n_s
\end{bmatrix}.
$$

Матрица системы, совпадающая с используемой в `_return_mapping_stress`, равна

$$
\mathbf H_{ts}
=
\begin{bmatrix}
E_n+H_t & E_nb \\
E_na & G_s+E_nab
\end{bmatrix}.
$$

Множители находятся из

$$
\mathbf H_{ts}
\begin{bmatrix}
\Delta\lambda_t \\
\Delta\lambda_s
\end{bmatrix}
=
\begin{bmatrix}
F_t^{\mathrm{tr}} \\
F_s^{\mathrm{tr}}
\end{bmatrix}.
$$

Jacobian:

$$
\boxed{
\mathbf P_J^{ts}
=
\mathbf I
-
\mathbf C\mathbf M_{ts}
\mathbf H_{ts}^{-1}
\mathbf N_{ts}^{\mathsf T}
}.
$$

Поскольку $c(q)$ в текущем return mapping вычисляется по $q_{\mathrm{old}}$, в элементе второй строки и первого столбца нет добавки, связанной с $dc/dq_{\mathrm{new}}$.

### 5.8 Угол сжатие–сдвиг `c+s`

Для $\mathcal A=\{c,s\}$

$$
\mathbf M_{cs}
=
\begin{bmatrix}
\mathbf m_c & \mathbf m_s
\end{bmatrix},
\qquad
\mathbf N_{cs}
=
\begin{bmatrix}
\mathbf n_c & \mathbf n_s
\end{bmatrix}.
$$

Матрица системы:

$$
\mathbf H_{cs}
=
\begin{bmatrix}
E_n & -E_nb \\
-E_na & G_s+E_nab
\end{bmatrix}.
$$

$$
\mathbf H_{cs}
\begin{bmatrix}
\Delta\lambda_c \\
\Delta\lambda_s
\end{bmatrix}
=
\begin{bmatrix}
F_c^{\mathrm{tr}} \\
F_s^{\mathrm{tr}}
\end{bmatrix},
$$

$$
\boxed{
\mathbf P_J^{cs}
=
\mathbf I
-
\mathbf C\mathbf M_{cs}
\mathbf H_{cs}^{-1}
\mathbf N_{cs}^{\mathsf T}
}.
$$

### 5.9 Проверка знаков множителей

После решения угловой системы код проверяет знаки множителей. Если один из них отрицателен, угловой набор отбрасывается и используется соответствующая одиночная грань. Согласованный Jacobian должен вычисляться для окончательно принятого набора, а не для первоначально предположенного угла.

---

## 6. Производные пластических множителей и упрочнения

Введём полную матрицу

$$
\mathbf L
=
\frac{\partial
[\Delta\lambda_t,\Delta\lambda_c,\Delta\lambda_s]^{\mathsf T}}
{\partial\boldsymbol\sigma^{\mathrm{tr}}}.
$$

В ней строки неактивных механизмов равны нулю, а активные строки берутся из $\mathbf L_{\mathcal A}$.

Обозначим строки этой матрицы как

$$
\boldsymbol\ell_t^{\mathsf T}=\mathbf L_{0:},
\qquad
\boldsymbol\ell_c^{\mathsf T}=\mathbf L_{1:},
\qquad
\boldsymbol\ell_s^{\mathsf T}=\mathbf L_{2:}.
$$

Тогда

$$
d\Delta\lambda_i
=
\boldsymbol\ell_i^{\mathsf T}
\,d\boldsymbol\sigma^{\mathrm{tr}}.
$$

Из закона упрочнения следует

$$
\boxed{
\mathbf g_q^{\mathsf T}
=
\frac{\partial q_{\mathrm{new}}}
     {\partial\boldsymbol\sigma^{\mathrm{tr}}}
=
H_t\boldsymbol\ell_t^{\mathsf T}
}.
$$

Обозначим первую и третью строки $\mathbf P_J$ как

$$
\mathbf p_n^{\mathsf T}=(\mathbf P_J)_{0:},
\qquad
\mathbf p_\tau^{\mathsf T}=(\mathbf P_J)_{2:}.
$$

Тогда

$$
d\widetilde\sigma_n
=
\mathbf p_n^{\mathsf T}d\boldsymbol\sigma^{\mathrm{tr}},
$$

$$
d\widetilde\tau
=
\mathbf p_\tau^{\mathsf T}d\boldsymbol\sigma^{\mathrm{tr}}.
$$

---

## 7. Производные пластической работы

Накопленные работы определяются как старое значение плюс неотрицательное приращение:

$$
W_i=W_{i,\mathrm{old}}+\max(\widehat{\Delta W}_i,0).
$$

Ниже сначала приведены градиенты внутреннего выражения $\widehat{\Delta W}_i$. Если ветвь `max` выбрала ноль и $\widehat{\Delta W}_i<0$, истинный градиент $W_i$ равен нулю. В точке $\widehat{\Delta W}_i=0$ производная не единственна.

### 7.1 Работа растяжения

В коде

$$
\widehat{\Delta W}_t
=
\frac12
\left(f_t+q_{\mathrm{old}}+\widetilde\sigma_n\right)
\Delta\lambda_t.
$$

Следовательно,

$$
\boxed{
\mathbf g_{W_t}^{\mathsf T}
=
\frac12\Delta\lambda_t\mathbf p_n^{\mathsf T}
+
\frac12
\left(f_t+q_{\mathrm{old}}+\widetilde\sigma_n\right)
\boldsymbol\ell_t^{\mathsf T}
}.
$$

### 7.2 Работа сжатия

$$
\widehat{\Delta W}_c
=
|\widetilde\sigma_n|\Delta\lambda_c.
$$

При $\widetilde\sigma_n\ne0$

$$
\boxed{
\mathbf g_{W_c}^{\mathsf T}
=
\Delta\lambda_c
\operatorname{sign}(\widetilde\sigma_n)
\mathbf p_n^{\mathsf T}
+
|\widetilde\sigma_n|
\boldsymbol\ell_c^{\mathsf T}
}.
$$

### 7.3 Работа сдвига

$$
\widehat{\Delta W}_s
=
\left(
|\widetilde\tau|+b\widetilde\sigma_n
\right)
\Delta\lambda_s.
$$

При $\widetilde\tau\ne0$

$$
\boxed{
\mathbf g_{W_s}^{\mathsf T}
=
\Delta\lambda_s
\left[
\operatorname{sign}(\widetilde\tau)
\mathbf p_\tau^{\mathsf T}
+b\mathbf p_n^{\mathsf T}
\right]
+
\left(
|\widetilde\tau|+b\widetilde\sigma_n
\right)
\boldsymbol\ell_s^{\mathsf T}
}.
$$

Для неактивного механизма его множитель и соответствующая строка $\mathbf L$ равны нулю.

---

## 8. Производные повреждённости

### 8.1 Нормированные энергии и saturation

Определим

$$
r_t=\min\!\left(\frac{W_t}{G_{f,t}},1\right),
\qquad
r_c=\min\!\left(\frac{W_c}{G_{f,c}},1\right),
\qquad
r_s=\min\!\left(\frac{W_s}{G_{f,s}},1\right).
$$

На ненасыщенной ветви $W_i/G_{f,i}<1$

$$
\mathbf g_{r_i}^{\mathsf T}
=
\frac{\mathbf g_{W_i}^{\mathsf T}}{G_{f,i}}.
$$

На насыщенной ветви $W_i/G_{f,i}>1$

$$
\mathbf g_{r_i}^{\mathsf T}=\mathbf0.
$$

При $W_i/G_{f,i}=1$ берётся односторонняя производная выбранной ветви.

Функции эволюции:

$$
F_t=r_t(2-r_t),
\qquad
F_s=r_s(2-r_s),
$$

$$
F_c
=
\frac12
\left[
\sin\!\left(\pi r_c-\frac\pi2\right)+1
\right].
$$

На ненасыщенной ветви

$$
\mathbf g_{F_t}^{\mathsf T}
=
2(1-r_t)\mathbf g_{r_t}^{\mathsf T},
$$

$$
\mathbf g_{F_s}^{\mathsf T}
=
2(1-r_s)\mathbf g_{r_s}^{\mathsf T},
$$

$$
\mathbf g_{F_c}^{\mathsf T}
=
\frac\pi2
\cos\!\left(\pi r_c-\frac\pi2\right)
\mathbf g_{r_c}^{\mathsf T}.
$$

На насыщенной ветви значения функций остаются равными единице, а их градиенты равны нулю.

### 8.2 Растягивающее нормальное повреждение

Промежуточная функция

$$
d_t
=
F_t+a_tF_s(1-F_t).
$$

Её градиент:

$$
\mathbf g_{d_t}^{\mathsf T}
=
(1-a_tF_s)\mathbf g_{F_t}^{\mathsf T}
+
a_t(1-F_t)\mathbf g_{F_s}^{\mathsf T}.
$$

Пусть

$$
L_t=f_t+q_{\mathrm{new}}+\epsilon_q,
$$

где в коде $\epsilon_q=10^{-12}$. Тогда

$$
D_{nt}^{\mathrm{raw}}
=
1-(1-d_t)\frac{f_t}{L_t}.
$$

Дифференцирование даёт

$$
\boxed{
\mathbf g_{D_{nt}^{\mathrm{raw}}}^{\mathsf T}
=
\frac{f_t}{L_t}\mathbf g_{d_t}^{\mathsf T}
+
(1-d_t)\frac{f_t}{L_t^2}
\mathbf g_q^{\mathsf T}
}.
$$

### 8.3 Сжимающее нормальное повреждение

$$
D_{nc}^{\mathrm{raw}}
=
\left(1-\frac{f_{c,r}}{f_c}\right)F_c.
$$

В коде отношение $f_{c,r}/f_c$ хранится как `fcr_over_fc`. Поэтому

$$
\boxed{
\mathbf g_{D_{nc}^{\mathrm{raw}}}^{\mathsf T}
=
\left(1-\frac{f_{c,r}}{f_c}\right)
\mathbf g_{F_c}^{\mathsf T}
}.
$$

### 8.4 Базовое сдвиговое повреждение

До ограничения в интервале $[0,1]$ код вычисляет

$$
B
=
a_sF_t(1-F_s)(1-F_c)
+F_s+F_c-F_sF_c.
$$

Его градиент:

$$
\begin{aligned}
\mathbf g_B^{\mathsf T}
={}&
a_s(1-F_s)(1-F_c)
\mathbf g_{F_t}^{\mathsf T}
\\
&+
\left[
1-F_c-a_sF_t(1-F_c)
\right]
\mathbf g_{F_s}^{\mathsf T}
\\
&+
\left[
1-F_s-a_sF_t(1-F_s)
\right]
\mathbf g_{F_c}^{\mathsf T}.
\end{aligned}
$$

После этого применяется

$$
B_{\mathrm{clip}}
=
\min(\max(B,0),1).
$$

На внутренней ветви $0<B<1$

$$
\mathbf g_{B_{\mathrm{clip}}}^{\mathsf T}
=
\mathbf g_B^{\mathsf T}.
$$

На ветвях $B<0$ или $B>1$

$$
\mathbf g_{B_{\mathrm{clip}}}^{\mathsf T}
=
\mathbf0.
$$

### 8.5 Сдвиговое повреждение при растяжении и сжатии

При $\widetilde\sigma_n\ge0$

$$
D_s^{\mathrm{raw}}=B_{\mathrm{clip}},
$$

$$
\mathbf g_{D_s^{\mathrm{raw}}}^{\mathsf T}
=
\mathbf g_{B_{\mathrm{clip}}}^{\mathsf T}.
$$

При $\widetilde\sigma_n<0$ положим

$$
x=-\widetilde\sigma_n,
\qquad
a_r=\tan\phi_r.
$$

Коэффициент остаточного трения:

$$
R
=
\frac{c+x(a-a_r)}{c+xa+\epsilon_c},
$$

где в коде $\epsilon_c=10^{-12}$. Тогда

$$
D_s^{\mathrm{raw}}
=
B_{\mathrm{clip}}R.
$$

Если пренебречь исключительно регуляризующей добавкой $\epsilon_c$ при упрощении числителя, производная коэффициента по нормальному эффективному напряжению равна

$$
\frac{\partial R}{\partial\widetilde\sigma_n}
\approx
\frac{ca_r}{(c+xa+\epsilon_c)^2}.
$$

Именно эта форма используется в `_joint_jacobian`. Поэтому

$$
\boxed{
\mathbf g_{D_s^{\mathrm{raw}}}^{\mathsf T}
=
R\mathbf g_{B_{\mathrm{clip}}}^{\mathsf T}
+
B_{\mathrm{clip}}
\frac{ca_r}{(c+xa+\epsilon_c)^2}
\mathbf p_n^{\mathsf T}
}.
$$

Если дифференцировать регуляризованную дробь буквально, числитель производной содержит также члены порядка $\epsilon_c$; production-код их не учитывает ввиду их пренебрежимо малого масштаба.

### 8.6 Верхние caps и необратимость

Сначала `_calculate_damage` применяет верхние ограничения

$$
D_i^{\mathrm{cap}}
=
\min(D_i^{\mathrm{raw}},D_i^{\max}).
$$

Затем `_evaluate_joint` обеспечивает необратимость:

$$
D_i
=
\max(D_i^{\mathrm{cap}},D_{i,\mathrm{old}}).
$$

Внутри свободной ветви

$$
D_{i,\mathrm{old}}<D_i^{\mathrm{raw}}<D_i^{\max}
$$

градиент совпадает с градиентом исходной функции. На ветви необратимости или верхнего cap

$$
\mathbf g_{D_i}^{\mathsf T}=\mathbf0.
$$

Это можно записать через индикатор активности $\zeta_i$:

$$
\mathbf g_{D_i}^{\mathsf T}
=
\zeta_i
\mathbf g_{D_i^{\mathrm{raw}}}^{\mathsf T},
$$

$$
\zeta_i
=
\begin{cases}
1, & D_{i,\mathrm{old}}<D_i^{\mathrm{raw}}<D_i^{\max},\\
0, & D_i^{\mathrm{raw}}<D_{i,\mathrm{old}},\\
0, & D_i^{\mathrm{raw}}>D_i^{\max}.
\end{cases}
$$

В production-коде на равенстве градиент также обнуляется посредством условий `<=` и `>=`.

### 8.7 Выбор нормальной ветви по знаку напряжения

Применяемое нормальное повреждение зависит от знака эффективного нормального напряжения:

$$
D_n
=
\begin{cases}
D_{nt}, & \widetilde\sigma_n\ge0,\\
D_{nc}, & \widetilde\sigma_n<0.
\end{cases}
$$

Соответственно,

$$
\mathbf g_{D_n}^{\mathsf T}
=
\begin{cases}
\mathbf g_{D_{nt}}^{\mathsf T}, & \widetilde\sigma_n\ge0,\\
\mathbf g_{D_{nc}}^{\mathsf T}, & \widetilde\sigma_n<0.
\end{cases}
$$

При $\widetilde\sigma_n=0$ обычная двусторонняя производная может отсутствовать.

### 8.8 Повреждение компоненты вдоль шва

Если `matrix_parallel_damage_cap == 0`, код не повреждает вторую локальную компоненту:

$$
D_p=0,
\qquad
\mathbf g_{D_p}^{\mathsf T}=\mathbf0.
$$

Если `matrix_parallel_damage_cap > 0`, то

$$
D_p
=
\min\!\left(
\max(D_n,D_s),
D_{p,\max}
\right).
$$

До достижения cap градиент берётся от большей переменной:

$$
\mathbf g_{D_p}^{\mathsf T}
=
\begin{cases}
\mathbf g_{D_n}^{\mathsf T}, & D_n\ge D_s,\\
\mathbf g_{D_s}^{\mathsf T}, & D_s>D_n.
\end{cases}
$$

После достижения cap

$$
\mathbf g_{D_p}^{\mathsf T}=\mathbf0.
$$

При $D_n=D_s$ максимум недифференцируем; код односторонне выбирает ветвь $D_n$.

---

## 9. Полный локальный Jacobian: пластичность и damage

Диагональная матрица сохранённой доли жёсткости/напряжения:

$$
\mathbf G_D
=
\operatorname{diag}
\left(
1-D_n,
1-D_p,
1-D_s
\right).
$$

Номинальное локальное напряжение:

$$
\boldsymbol\sigma_{\mathrm{nom}}^l
=
\mathbf G_D
\widetilde{\boldsymbol\sigma}^{l}.
$$

Соберём градиенты повреждённости по локальному пробному напряжению в матрицу

$$
\mathbf B_D
=
\begin{bmatrix}
\mathbf g_{D_n}^{\mathsf T} \\
\mathbf g_{D_p}^{\mathsf T} \\
\mathbf g_{D_s}^{\mathsf T}
\end{bmatrix}.
$$

Дифференцируя произведение, получаем

$$
\begin{aligned}
d\boldsymbol\sigma_{\mathrm{nom}}^l
={}&
\mathbf G_D
\,d\widetilde{\boldsymbol\sigma}^l
-
\operatorname{diag}
\left(
\widetilde{\boldsymbol\sigma}^l
\right)
\,d\mathbf D,
\end{aligned}
$$

где

$$
d\widetilde{\boldsymbol\sigma}^l
=
\mathbf P_J
\,d\boldsymbol\sigma_{\mathrm{tr}}^l,
$$

$$
d\mathbf D
=
\mathbf B_D
\,d\boldsymbol\sigma_{\mathrm{tr}}^l.
$$

Итак, полный Jacobian шва равен

$$
\boxed{
\mathbf J_J
=
\mathbf G_D\mathbf P_J
-
\operatorname{diag}
\left(
\widetilde{\boldsymbol\sigma}^l
\right)
\mathbf B_D
}.
$$

Первое слагаемое учитывает return mapping и текущее значение повреждённости. Второе учитывает изменение самой повреждённости при изменении деформации. Без второго слагаемого оператор не является согласованной производной на активной ветви damage.

---

## 10. Финальная глобальная матрица

Подстановка локального Jacobian в цепное правило даёт

$$
\boxed{
\mathbf D_{\mathrm{alg}}^g
=
\mathbf T_\varepsilon^{\mathsf T}
\left[
\mathbf G_D\mathbf P_J
-
\operatorname{diag}
\left(
\widetilde{\boldsymbol\sigma}^l
\right)
\mathbf B_D
\right]
\mathbf T_\sigma
\mathbf A_m
\mathbf D_r
}.
$$

Каждый множитель имеет самостоятельный смысл:

- $\mathbf D_r$ переводит приращение глобальной деформации в пробное эффективное напряжение матрицы;
- $\mathbf A_m$ линеаризует return mapping Друкера—Прагера;
- $\mathbf T_\sigma$ переводит исправленное напряжение матрицы в локальные оси шва;
- $\mathbf P_J$ линеаризует многоповерхностный return mapping шва;
- $\mathbf G_D$ применяет текущее ослабление;
- $\mathbf B_D$ учитывает эволюцию damage;
- $\mathbf T_\varepsilon^{\mathsf T}$ возвращает локальное номинальное напряжение в глобальные оси.

Матрица в общем случае несимметрична. Причины:

1. неассоциированное течение матрицы при $\psi_m\ne\phi_m$;
2. неассоциированное сдвиговое течение шва при $\psi\ne\phi$;
3. направленная зависимость пластической работы и повреждённости;
4. последовательная композиция нелинейных отображений.

Принудительная симметризация изменяет точный Jacobian алгоритма и может ухудшить квадратичную сходимость глобального метода Ньютона.

---

## 11. Соответствие текущему production-коду

Текущая реализация использует следующие методы:

- `_return_mapping_matrix_with_jacobian` возвращает $\boldsymbol\sigma_m^g$ и $\mathbf A_m$;
- `_return_mapping_stress` возвращает $\widetilde{\boldsymbol\sigma}^l$, множители и приращение пластической деформации;
- `_joint_jacobian` строит $\mathbf P_J$, производные работ, damage и полный $\mathbf J_J$;
- `_evaluate_joint` вычисляет напряжение и передаёт в `update_state` локальный Jacobian;
- `update_state` собирает глобальное цепное произведение.

Сцепление в return mapping шва вычисляется по $q_{\mathrm{old}}$. Поэтому приведённый Jacobian корректно не содержит производную $dc(q_{\mathrm{new}})/dq$.

Функция `_compute_secant_matrix` остаётся в классе как отдельная секущая аппроксимация, но текущий `update_state` после активации шва её не использует для `D_tangent`.

### 11.1 Технические особенности текущего `_joint_jacobian`

Для академической точности необходимо отметить два узких случая текущей реализации:

1. В локальной функции `ratio_grad` при $W/G_f\ge1$ возвращаются нулевой градиент и нулевое вспомогательное значение функции эволюции. Математически при saturation значение $F_i$ должно оставаться равным единице, а нулевым становится только градиент. Поэтому на полностью насыщенной смешанной ветви текущий код может неверно вычислить перекрёстные производные других damage-переменных, хотя само напряжение использует правильно рассчитанные значения $D_i$ из `_calculate_damage`.
2. Приращения работ в `_evaluate_joint` ограничиваются через `max(raw, 0)`, но `_joint_jacobian` явно не обнуляет соответствующий градиент при отрицательном `raw`. На стандартных допустимых активных ветвях проверенных тестов выражения неотрицательны; для общего случая точная производная clipped-ветви должна следовать правилам раздела 7.

Эти особенности не изменяют аналитический вывод истинной производной дискретного закона напряжений, но должны учитываться при расширении тестов на насыщенные и пограничные режимы.

---

## 12. Точки недифференцируемости

Обычная классическая производная может отсутствовать в следующих состояниях:

- первое образование плоскости и дискретный выбор критической нормали;
- смена активного набора `elastic/t/c/s/t+s/c+s`;
- переход пластического множителя через ноль;
- $\tau^{\mathrm{tr}}=0$ или $\widetilde\tau=0$ из-за производной модуля;
- $\widetilde\sigma_n=0$ при переключении $D_{nt}\leftrightarrow D_{nc}$;
- $q=q_{\mathrm{lim}}$ в кусочном законе сцепления;
- $W_i/G_{f,i}=1$ при saturation;
- $B=0$ и $B=1$ при clipping базового сдвигового damage;
- $D_i^{\mathrm{raw}}=D_{i,\mathrm{old}}$ при включении необратимости;
- $D_i^{\mathrm{raw}}=D_i^{\max}$ при достижении cap;
- $D_n=D_s$ в операции максимума для $D_p$;
- $\max(D_n,D_s)=D_{p,\max}$;
- граница smooth/apex return матрицы;
- вершина $p=p_{\mathrm{apex}}$.

Практический алгоритм использует Jacobian фактически принятой односторонней ветви. При поиске критической плоскости ориентация не дифференцируется: после выбора она должна рассматриваться как фиксированная в локальной линеаризации.

---

## 13. Пошаговый алгоритм реализации

Для каждого вызова `update_state` согласованная процедура имеет следующий вид.

### Шаг 1. Упругий прогноз матрицы

$$
\boldsymbol\sigma_{\mathrm{tr}}^g
=
\boldsymbol\sigma_{\mathrm{eff,old}}^g
+
\mathbf D_r
\left(
\boldsymbol\varepsilon^g-
\boldsymbol\varepsilon_{\mathrm{old}}^g
\right).
$$

### Шаг 2. Return mapping матрицы

1. Вычислить $p$, $s$, $q$ и $F_m^{\mathrm{tr}}$.
2. Выбрать ветвь elastic, smooth или apex.
3. Получить $\boldsymbol\sigma_m^g$.
4. Собрать $\mathbf A_m$ для той же ветви.

### Шаг 3. Проверка или фиксация шва

Если шов отсутствует и критерии не нарушены, вернуть

$$
\boldsymbol\sigma^g=\boldsymbol\sigma_m^g,
\qquad
\mathbf D_{\mathrm{alg}}^g=\mathbf A_m\mathbf D_r.
$$

Если плоскость задана или активирована, построить $\mathbf T_\sigma$, $\mathbf T_\varepsilon$ и $\mathbf C$ и далее считать ориентацию фиксированной.

### Шаг 4. Локальное пробное напряжение

$$
\boldsymbol\sigma_{\mathrm{tr}}^l
=
\mathbf T_\sigma\boldsymbol\sigma_m^g.
$$

### Шаг 5. Active-set return mapping шва

1. Вычислить $F_t^{\mathrm{tr}}$, $F_c^{\mathrm{tr}}$, $F_s^{\mathrm{tr}}$.
2. Выбрать предполагаемый активный набор.
3. Решить одиночное уравнение или систему $2\times2$.
4. Проверить неотрицательность множителей.
5. При отрицательном множителе перейти на соответствующую грань.
6. Получить $\widetilde{\boldsymbol\sigma}^l$ и окончательный active set.

### Шаг 6. Jacobian пластичности шва

1. Собрать $\mathbf M_{\mathcal A}$ и $\mathbf N_{\mathcal A}$.
2. Собрать $\mathbf H_{\mathcal A}$.
3. Вычислить $\mathbf L_{\mathcal A}$.
4. Встроить активные строки в полную матрицу $\mathbf L$.
5. Вычислить $\mathbf P_J$.

### Шаг 7. Работы и упрочнение

1. Обновить $q_{\mathrm{new}}$.
2. Вычислить $W_t$, $W_c$, $W_s$.
3. Собрать $\mathbf g_q$, $\mathbf g_{W_t}$, $\mathbf g_{W_c}$, $\mathbf g_{W_s}$.
4. Учесть активные ветви `max(raw, 0)`.

### Шаг 8. Damage

1. Вычислить $r_i$ и определить saturation-ветви.
2. Вычислить $F_t$, $F_c$, $F_s$ и их градиенты.
3. Вычислить $D_{nt}$, $D_{nc}$, $D_s$.
4. Применить clipping, caps и необратимость вместе с соответствующим обнулением градиентов.
5. Выбрать $D_n$ по знаку $\widetilde\sigma_n$.
6. При необходимости вычислить $D_p$.
7. Собрать $\mathbf G_D$ и $\mathbf B_D$.

### Шаг 9. Локальный полный Jacobian

$$
\mathbf J_J
=
\mathbf G_D\mathbf P_J
-
\operatorname{diag}
\left(
\widetilde{\boldsymbol\sigma}^l
\right)
\mathbf B_D.
$$

### Шаг 10. Глобальный оператор

$$
\mathbf D_{\mathrm{alg}}^g
=
\mathbf T_\varepsilon^{\mathsf T}
\mathbf J_J
\mathbf T_\sigma
\mathbf A_m
\mathbf D_r.
$$

---

## 14. Проверка центральными конечными разностями

Аналитический оператор следует проверять на каждой гладкой ветви отдельно. При фиксированном committed-состоянии и базовой деформации $\boldsymbol\varepsilon$ численная производная столбца $j$ вычисляется как

$$
\left(\mathbf D_{\mathrm{FD}}\right)_{:j}
=
\frac{
\boldsymbol\sigma
\left(
\boldsymbol\varepsilon+h_j\mathbf e_j
\right)
-
\boldsymbol\sigma
\left(
\boldsymbol\varepsilon-h_j\mathbf e_j
\right)
}{2h_j}.
$$

Для корректной проверки необходимо:

1. создавать идентичное исходное состояние для вычислений `+h` и `-h`;
2. не переносить trial-состояние между возмущениями;
3. фиксировать ориентацию шва;
4. выбирать $h_j$ достаточно малым для локальной производной, но не настолько малым, чтобы доминировала ошибка округления;
5. не допускать смены active set между `+h` и `-h`;
6. отдельно проверять точки вблизи переключений односторонними разностями.

Сравнение выполняется по норме, например

$$
\frac{
\lVert
\mathbf D_{\mathrm{alg}}-
\mathbf D_{\mathrm{FD}}
\rVert
}{
\max(1,\lVert\mathbf D_{\mathrm{FD}}\rVert)
}
<\mathrm{tol}.
$$

Минимальный набор проверок должен включать:

- упругую матрицу без шва;
- гладкий return матрицы;
- apex матрицы;
- упругий шов;
- чистое растяжение `t`;
- чистое сжатие `c`;
- чистый сдвиг `s` для обоих знаков $\tau$;
- углы `t+s` и `c+s`;
- активную эволюцию $D_{nt}$;
- активную эволюцию $D_{nc}$;
- активную эволюцию $D_s$ при растяжении и сжатии;
- saturation;
- необратимость;
- каждый верхний cap;
- включённое повреждение $D_p$ вдоль шва.

Существующий файл
`MyTesting/test_algorithmic_tangent_ubiquitous_joint_2d.py`
проверяет основные гладкие ветви сравнением с центральными конечными разностями.

---

## 15. Итог

Для фактической последовательной схемы `matrix → joint → damage` алгоритмически согласованная глобальная матрица не является ни суммой матриц механизмов, ни простой повреждённой секущей матрицей. Она является произведением Jacobian-ов дискретных отображений:

$$
\boxed{
\mathbf D_{\mathrm{alg}}^g
=
\mathbf T_\varepsilon^{\mathsf T}
\left[
\mathbf G_D\mathbf P_J
-
\operatorname{diag}
\left(
\widetilde{\boldsymbol\sigma}^l
\right)
\mathbf B_D
\right]
\mathbf T_\sigma
\mathbf A_m
\mathbf D_r
}.
$$

Эта формула одновременно учитывает:

- elastic, smooth и apex-ветви матрицы Друкера—Прагера;
- все активные наборы шва;
- неассоциированное течение;
- линейное упрочнение растягивающей поверхности;
- эволюцию пластической работы;
- анизотропную повреждённость;
- saturation, irreversibility и caps;
- корректное преобразование между глобальными и локальными координатами.

Вне точек переключения именно эта матрица является производной напряжения, возвращаемого `update_state`, по входной глобальной деформации.
