# Agent-based population evolution simulator with neural network guidance

Симуляция эволюции популяции в изменяющейся среде. Каждая особь описывается вектором признаков (температурная устойчивость, скорость, размер и др.). Функция приспособленности определяет выживаемость в текущих условиях среды.

Популяция клонируется на две параллельные ветви. Одна эволюционирует по классическому генетическому алгоритму (baseline), другая получает вектор направления мутаций от обученного `CatBoost`-советчика (neural-guided). По итогу каждого поколения сравниваются численность, средний fitness и скорость адаптации.

---

## Команда

| Участник | Уникальная библиотека | Направление |
|---|---|---|
| Левенец Даниэль | `mesa` | Ядро симуляции, генетический алгоритм, тесты, профилирование |
| Шевченко Никита | `catboost` | Нейросоветчик (CatBoost), визуализация |

---

## Архитектура

```
evosim/
├── core/
│   ├── model.py             # Mesa-модель, главный цикл, DataCollector
│   ├── population.py        # популяция: инициализация, отбор, размножение, конкуренция
│   ├── individual.py        # агент-особь: геном, возраст, смертность
│   ├── environment.py       # среда: температура, пища, распределение ресурса
│   ├── fitness.py           # функции фитнеса (temp / energy / hazard / proportion)
│   ├── mutation_patterns.py # стратегии мутации ядра симуляции
│   ├── recording.py         # protocol для внешнего recorder без знания о ML
│   └── enums.py             # DeathCause, Gender
├── neural/
│   ├── comparison_runner.py # orchestration baseline vs ML-guided comparison
│   ├── guided_mutation.py   # guided mutation strategy for comparison branch
│   ├── training_buffer.py   # сбор (x, y, weights) для обучения советчика
│   ├── advisor.py           # CatBoostAdvisor: обёртка над CatBoostRegressor
│   ├── trainer.py           # обучение советчика на буфере baseline
│   └── mutation.py          # neural_mutate: сдвиг генома к предсказанному вектору
├── vis/
│   ├── history.py           # пошаговый сбор истории → DataFrame
│   ├── plots.py             # Plotly-графики и компоненты дашборда
│   ├── snapshots.py         # per-step снимки геномов для анимации
│   ├── dashboard.py         # сборка сравнительного HTML-дашборда baseline vs neural
│   └── vis_demo.py          # демо базовой визуализации (одиночный прогон)
├── profiling/
│   ├── profiler.py          # cProfile + сохранение артефактов
│   ├── stats.py             # фильтрация project-only функций, текстовая сводка
│   ├── report.py            # рендер HTML-отчёта
│   └── report_template.html
├── tests/                   # pytest: fitness / individual / population / environment /
│                            # training_buffer / advisor / trainer / mutation
├── reports/                 # артефакты профилирования
├── config/
│   ├── sim_config.py        # dataclass-схемы (Sim / Comparison / Profiler + подконфиги)
│   ├── scenarios.py         # пресеты EnvironmentConfig
│   ├── registry.py          # регистрация схем и сценариев в Hydra ConfigStore
│   ├── sim_config.yaml      # дефолты для main.py
│   ├── compare_config.yaml  # дефолты для comparison_run.py
│   └── profile_config.yaml  # дефолты для profile_run.py
├── main.py                  # основной запуск чистой симуляции
├── comparison_run.py        # baseline + neural-guided comparison pipeline
├── profile_run.py           # точка входа профайлера
├── tests_run.py             # запуск всего тестового набора
├── requirements.txt
└── README.md
```

---

### Итерация 1

**Левенец**
- **1.1** Эволюционный движок через `mesa`
- **1.2** Функция приспособленности — `fitness(individual, environment)`. Определяет рождаемость и смертность.
- **1.3** CLI — параметры среды, число поколений

**Шевченко**
- **1.4** Cоветчик (CatBoost) — сеть предсказывает оптимальный вектор признаков. Обучается на данных первых поколений baseline
- **1.5** Интеграция советчика — `neural_mutate()` сдвигает признаки к целевому вектору вместо случайного шума
- **1.6** Базовая визуализация Plotly — графики численности, среднего fitness и средних признаков на одном полотне

---

### Итерация 2

**Левенец**
- **2.1** Тесты (pytest) — покрытие `fitness`, `individual`, `population`, `environment`, `training_buffer`
- **2.2** Отчёт по профилированию — `cProfile` на эволюционном цикле, HTML- и текстовая сводки
- **2.3** Расширенные сценарии среды
- **2.4** Параллельные ветви — comparison runner клонирует конфигурацию на baseline и neural-guided, единый интерфейс истории обеих ветвей

**Шевченко**
- **2.4** Переобучение модели каждые K поколений — советчик обновляется на свежей истории и адаптируется к изменению среды, а не только к начальным условиям
- **2.5** Сравнительный дашборд Plotly — анимация по поколениям: точки особей в пространстве признаков
- **2.6** Итоговая таблица baseline vs neural — численность, средний fitness, скорость адаптации

---

## Сборка

```bash
git clone https://github.com/levenetsdaniel/python-evolution-simulation
cd evosim

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Точки входа

| Скрипт | Что делает |
|---|---|
| `python3 main.py` | Основной прогон чистой симуляции. С `view=true` сохраняет базовый набор графиков по одной ветке; с `model_info`, `population_info`, `individual_info` печатает диагностические сводки. |
| `python3 comparison_run.py` | ML-guided comparison pipeline: обучает советчика на baseline-буфере и запускает обе ветви (baseline + neural-guided). С `view=true` собирает сравнительный дашборд. |
| `python3 profile_run.py` | Прогон симуляции под `cProfile` |
| `python3 vis/vis_demo.py` | Сохраняет базовый набор Plotly-графиков по одиночному прогону |
| `python3 vis/dashboard.py` | Прямой сбор comparison dashboard с дефолтным `ComparisonConfig` |
| `python3 tests_run.py` | Запуск всего тестового набора |

---

## Конфигурация

Все параметры описаны как dataclass-схемы в `config/sim_config.py` и зарегистрированы в Hydra ConfigStore (`config/registry.py`). Дефолтные значения берутся из dataclass; YAML-файлы `config/sim_config.yaml`, `config/compare_config.yaml` и `config/profile_config.yaml` задают стартовые профили; всё остальное переопределяется из CLI в стиле Hydra:

```bash
python3 main.py n_steps=200 seed=7
python3 main.py population.initial_size=300 environment.temp_start=10
python3 main.py environment=warming        # выбор сценария (см. ниже)
python3 main.py --help                     # полный список параметров
```

### `SimConfig` (корень)

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `n_steps` | int | `600` | Число шагов симуляции |
| `seed` | int | `42` | Сид генератора случайных чисел |
| `view` | bool | `false` | В `main.py` — собрать и сохранить базовые графики по одной ветке. В `profile_run.py` — автоматически открыть HTML-отчёт по профилированию. |
| `debug` | bool | `false` | Печатать подробную статистику по каждому шагу |
| `model_info` | bool | `false` | Сводка по модели (`main.py`) |
| `steps_info` | int | `5` | Сколько последних шагов показать в `model_info` |
| `population_info` | bool | `false` | Сводка по популяции (`main.py`) |
| `individual_info` | bool | `false` | Сводка по агентам (`main.py`) |

### `ComparisonConfig`

Используется только в `comparison_run.py` и для comparison dashboard.

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `simulation` | `SimConfig` | дефолты | Базовый конфиг обеих веток comparison pipeline |
| `retrain_steps` | int | `50` | Период переобучения советчика по baseline-буферу |
| `shift_strength` | float | `0.7` | Сила сдвига генома в guided mutation |
| `x_trait` | str | `heat_resistance` | Ось X для animated scatter comparison dashboard |
| `y_trait` | str | `cold_resistance` | Ось Y для animated scatter comparison dashboard |
| `view` | bool | `false` | В `comparison_run.py` — собрать и сохранить comparison dashboard вместо обычного прогона |

### `environment` (`EnvironmentConfig`)

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `food_availability` | float | `10000.0` | Пищевой ресурс на шаг |
| `food_step` | float | `0.0` | Изменение пищевой базы за шаг |
| `temp_start` | float | `20.0` | Стартовая температура (°C) |
| `temp_step` | float | `0.05` | Изменение температуры за шаг |
| `temp_reset` | float | `-5.0` | Значение, к которому сбрасывается температура при достижении границ |
| `hazard_level_start` | float | `0.1` | Стартовый уровень опасности |
| `hazard_step` | float | `0.001` | Изменение опасности за шаг |
| `min_temperature` | float | `-30.0` | Нижняя граница температуры |
| `max_temperature` | float | `50.0` | Верхняя граница температуры |

### `population` (`PopulationConfig`)

| Ключ | Тип | По умолчанию | Описание                                              |
|---|---|---|-------------------------------------------------------|
| `initial_size` | int | `100` | Начальный размер популяции                            |
| `mutation_std` | float | `0.12` | Стандартное отклонение baseline-мутации               |
| `reproduction_rate` | float | `0.4` | Коэффициент плодовитости                              |
| `min_reproduction_age` | int | `2` | Минимальный возраст для размножения                   |
| `wound_base` | float | `0.4` | Базовая вероятность смерти проигравшего в конкуренции |
| `genome_labels` | list[str] | 7 признаков | Названия генов                                        |

### `individual` (`IndividualConfig`)

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `fitness_death_threshold` | float | `0.1` | Порог fitness, ниже которого особь гибнет мгновенно |
| `fitness_death_prob_coef` | float | `0.1` | Множитель вероятности смерти от низкого fitness |
| `age_scale` | float | `80.0` | Масштаб возрастной смертности |
| `age_death_power` | float | `1.2` | Показатель степени в возрастной функции смертности |
| `food_need_size_coef` | float | `5.0` | Вклад `size` в потребность в пище |
| `food_need_resilience_coef` | float | `2.0` | Вклад `resilience` в потребность в пище |
| `food_need_speed_coef` | float | `1.0` | Вклад `speed` в потребность в пище |
| `food_need_aggr_coef` | float | `10.0` | Вклад `aggressiveness` в потребность в пище |

### `fitness` (`FitnessConfig`)

| Ключ | Тип | По умолчанию | Описание                                       |
|---|---|---|------------------------------------------------|
| `temp_20_norm` | float | `0.625` | Нормированная «комфортная» температура         |
| `temp_score_sharpness` | float | `0.5` | Жёсткость температурного штрафа                |
| `metabolic_rate_efficiency_penalty` | float | `0.5` | Штраф эффективности от метаболизма             |
| `resilience_efficiency_penalty` | float | `0.2` | Штраф эффективности от resilience              |
| `aggression_metabolic_penalty` | float | `0.1` | Штраф за избыток агрессии над метаболизмом     |
| `speed_metabolic_ratio` | float | `2.0` | Допустимое отношение `speed / metabolic_rate`  |
| `size_metabolic_ratio` | float | `1.5` | Допустимое отношение `size / metabolic_rate`   |
| `proportion_penalty` | float | `1.0` | Штраф за нарушение пропорций                   |
| `score_floor` | float | `0.05` | Минимальное значение любого компонента fitness |

---

## Сценарии

Готовые пресеты `EnvironmentConfig` определены в `config/scenarios.py` и подключаются через группу `environment` в CLI:

```bash
python3 main.py environment=warming
python3 main.py environment=famine population.initial_size=200 n_steps=1000
```

| Имя | Что моделирует | Что нагружает |
|---|---|---|
| `default` | Базовые значения | — |
| `stable` | Статичная среда, без климатического давления | Контрольный baseline |
| `warming` | Длительное монотонное потепление с холодного старта | `heat_resistance` |
| `ice_age` | Похолодание + сжатие пищевой базы | `cold_resistance`, метаболизм |
| `harsh_seasons` | Быстрые тепловые циклы (~150 шагов на цикл) | Двусторонняя терморегуляция |
| `famine` | Дефицит еды (≈4× ниже нормы), убывающая пищевая база | Конкуренция, `size`, метаболизм |
| `hazardous` | Уровень опасности растёт в 5× быстрее, старт выше | `resilience` |
| `chaos` | Сезонность + растущая опасность + дефицит еды одновременно | Стресс-тест |

---

## Примеры

Полный прогон симуляции:

```bash
python3 main.py
```

Сценарий + переопределение популяции:

```bash
python3 main.py environment=harsh_seasons population.initial_size=200 n_steps=1200
```

Одиночный baseline-прогон с печатью метрик модели:

```bash
python3 main.py n_steps=200 model_info=true
```

Сравнительный ML-guided запуск:

```bash
python3 comparison_run.py
python3 comparison_run.py simulation.environment=warming
python3 comparison_run.py retrain_steps=25 shift_strength=0.9
```

Полный дебаг-прогон с пошаговыми логами:

```bash
python3 main.py n_steps=50 debug=true population_info=true individual_info=true
```

Базовая визуализация (Plotly-графики в `vis/graphics/`):

```bash
python3 vis/vis_demo.py
```

---

## Сравнительный дашборд

```bash
python3 comparison_run.py view=true
python3 comparison_run.py view=true simulation.environment=warming simulation.seed=7
python3 comparison_run.py view=true simulation.environment=harsh_seasons simulation.population.initial_size=200 simulation.n_steps=1200
```
### Что внутри

- **Animated scatter (heat × cold).** Каждая точка — особь на двумерной проекции пространства признаков (по умолчанию `heat_resistance × cold_resistance`); цвет точки кодирует fitness (синяя гамма — baseline, красная — neural-guided). Внизу — кнопки play/pause и слайдер по поколениям для покадрового просмотра.
- **Comparison table.** Итоги двух ветвей по четырём метрикам:
  - финальная численность,
  - средний fitness в хвостовом окне (последние 50 шагов),
  - скорость адаптации — slope линейной аппроксимации `AvgFitness(generation)`,
  - сколько шагов реально прожили (до экстинкции либо до `n_steps`).

  В каждой строке лучшее значение подсвечивается зелёным.

### Ось проекции

В animated scatter оси задаются через параметры `x_trait` и `y_trait` в `ComparisonConfig` (по умолчанию — `heat_resistance × cold_resistance`). Любое имя из `population.genome_labels` валидно:

```bash
python3 comparison_run.py view=true x_trait=size y_trait=speed
python3 comparison_run.py view=true simulation.environment=warming x_trait=heat_resistance y_trait=metabolic_rate
```

---

## Профилирование

Профайлер прогоняет симуляцию под `cProfile` и собирает три артефакта в каталоге `reports/`:

| Файл | Назначение |
|---|---|
| `evosim.prof` | Бинарный дамп `cProfile` |
| `cprofile_summary.txt` | Текстовая сводка top-N по cumtime + tottime + callers |
| `profiling_report.html` | HTML-отчёт по коду проекта |

Импорты `mesa` / `numpy` / `pandas` / `scipy` / `catboost` в профилирование не включаются — фильтр в `profiling/stats.py` оставляет только функции из проекта.

### `ProfilerConfig`

Конфиг профайлера оборачивает `SimConfig` под ключом `simulation_config`. Параметры симуляции переопределяются через этот префикс:

```bash
python3 profile_run.py simulation_config.n_steps=500
python3 profile_run.py simulation_config.environment=warming
python3 profile_run.py simulation_config.population.initial_size=300 top_n_flame=25
```

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `simulation_config` | `SimConfig` | дефолты | Конфиг прогоняемой симуляции |
| `output_dir` | str | `reports` | Каталог для всех артефактов |
| `prof_filename` | str | `evosim.prof` | Имя `.prof` дампа |
| `text_summary_filename` | str | `cprofile_summary.txt` | Имя текстовой сводки |
| `html_report_filename` | str | `profiling_report.html` | Имя HTML-отчёта |
| `top_n_text` | int | `50` | Сколько строк включать в текстовую сводку |
| `top_n_callers` | int | `10` | Сколько горячих функций раскрывать в секции callers |
| `top_n_flame` | int | `15` | Сколько строк показывать во флеймчарте HTML |
| `top_n_console` | int | `15` | Сколько строк печатать в stdout по завершении |
| `view` | bool | `false` | Автоматически открыть HTML-отчёт в браузере |

### Примеры

Базовый прогон, 200 шагов:

```bash
python3 profile_run.py simulation_config.n_steps=200
```

С автооткрытием HTML-отчёта:

```bash
python3 profile_run.py simulation_config.n_steps=500 view=true
```

Углублённый анализ — больше callers и больше строк во флеймчарте:

```bash
python3 profile_run.py simulation_config.n_steps=500 top_n_callers=20 top_n_flame=25
```

Профилирование с увеличенным стартовым размером популяции:

```bash
python3 profile_run.py simulation_config.n_steps=1000 simulation_config.population.initial_size=300
```
