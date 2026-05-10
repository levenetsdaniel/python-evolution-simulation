# Agent-based population evolution simulator# Agent-based population evolution simulator with neural network guidance


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
│   ├── engine.py            # Engine: предобучение советчика + параллельные ветви
│   ├── model.py             # Mesa-модель, главный цикл, DataCollector
│   ├── population.py        # популяция: инициализация, отбор, размножение, конкуренция
│   ├── individual.py        # агент-особь: геном, возраст, смертность
│   ├── environment.py       # среда: температура, пища, распределение ресурса
│   ├── fitness.py           # функции фитнеса (temp / energy / hazard / proportion)
│   ├── mutation_patterns.py # стратегии мутации: BaselineMutation, NeuralMutation
│   ├── training_buffer.py   # сбор (x, y, weights) для обучения советчика
│   └── enums.py             # DeathCause, Gender
├── neural/
│   ├── advisor.py           # CatBoostAdvisor: обёртка над CatBoostRegressor
│   ├── trainer.py           # обучение советчика на буфере baseline
│   └── mutation.py          # neural_mutate: сдвиг генома к предсказанному вектору
├── vis/
│   ├── history.py           # пошаговый сбор истории → DataFrame
│   ├── plots.py             # Plotly-графики
│   └── vis_demo.py          # демо визуализации (одиночный прогон)
├── profiling/
│   ├── profiler.py          # cProfile + сохранение артефактов
│   ├── stats.py             # фильтрация project-only функций, текстовая сводка
│   ├── report.py            # рендер HTML-отчёта
│   └── report_template.html
├── tests/                   # pytest: fitness / individual / population / environment / training_buffer
├── reports/                 # артефакты профилирования
├── config/
│   ├── sim_config.py        # dataclass-конфиги (Sim / Environment / Population / Individual / Fitness / Profiler)
│   └── cli.py               # argparse → (SimConfig, ProfilerConfig)
├── main.py                  # baseline + neural-guided через Engine
├── synthetic_run.py         # одиночный Model: печать статистики, сбор обучающих примеров
├── profile_run.py           # точка входа профайлера
├── advisor_demo.py          # демо обучения и применения советчика
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
- **2.3** Расширенные сценарии среды — постепенное / резкое / циклическое изменение параметров
- **2.4** Параллельные ветви — `Engine` клонирует конфигурацию на baseline и neural-guided, единый интерфейс истории обеих ветвей

**Шевченко**
- **2.4** Переобучение модели каждые K поколений — советчик обновляется на свежей истории и адаптируется к изменению среды, а не только к начальным условиям
- **2.5** Сравнительный дашборд Plotly — анимация по поколениям: точки особей в пространстве признаков
- **2.6** Итоговая таблица baseline vs neural — численность, средний fitness, скорость адаптации

---

## Сборка

### Установка

```bash
git clone https://github.com/levenetsdaniel/python-evolution-simulation
cd evosim

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Точки входа

| Скрипт | Что делает |
|---|---|
| `python main.py` | Полный прогон `Engine`: предобучает советчика на baseline-буфере и запускает обе ветви (baseline + neural-guided) синхронно |
| `python synthetic_run.py` | Одиночный прогон `Model` (только baseline-мутации) с возможностью печати статистики и сохранения обучающего буфера |
| `python profile_run.py` | Прогон симуляции под `cProfile` |
| `python vis/vis_demo.py` | Сохраняет базовый набор Plotly-графиков по одиночному прогону |
| `python advisor_demo.py` | Мини-демо: обучение советчика, save/load, проверка корректности `neural_mutate` |
| `python tests_run.py` | Запуск всего тестового набора |

---

## CLI

Все параметры опциональны — по умолчанию используются значения из `config/sim_config.py`. Парсер общий для `main.py`, `synthetic_run.py` и `profile_run.py`.

### Симуляция

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--n-steps` | int | `600` | Число шагов симуляции |
| `--seed` | int | `42` | Сид генератора случайных чисел |
| `--shift-strength` | float | `0.7` | Сила сдвига генома к предсказанию советчика в `NeuralMutation` |
| `--output` | str | `data/training_samples.json` | Путь для сохранения собранных обучающих примеров (только `synthetic_run.py` при `--record`) |
| `--record` | flag | off | Записывать буфер обучающих примеров на диск (только `synthetic_run.py`) |
| `--debug` | flag | off | Печатать подробную статистику по каждому шагу |

### Вывод статистики (только `synthetic_run.py`)

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--model-info` | flag | off | Вывести сводку по модели |
| `--steps-info` | int | `5` | Сколько последних шагов показать для `--model-info` |
| `--population-info` | flag | off | Вывести метрики популяции |
| `--individual-info` | flag | off | Вывести метрики агентов |

### Среда (`EnvironmentConfig`)

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--food-availability` | float | `10000.0` | Пищевой ресурс на шаг |
| `--max-temp` | float | `50.0` | Верхняя граница температуры (°C) |
| `--min-temp` | float | `-30.0` | Нижняя граница температуры (°C) |

### Популяция (`PopulationConfig`)

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--population-size` | int | `100` | Начальный размер популяции |
| `--mutation-std` | float | `0.12` | Стандартное отклонение baseline-мутации |
| `--reproduction-rate` | float | `0.4` | Коэффициент плодовитости |
| `--min-reproduction-age` | int | `2` | Минимальный возраст для размножения |
| `--wound-base` | float | `0.4` | Базовая вероятность смерти проигравшего в конкуренции |

### Агент (`IndividualConfig`)

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--fitness-death-threshold` | float | `0.1` | Порог fitness, ниже которого особь гибнет мгновенно |
| `--fitness-death-prob-coef` | float | `0.1` | Множитель вероятности смерти от низкого fitness |
| `--age-scale` | float | `80.0` | Масштаб возрастной смертности (выше — дольше живут) |
| `--age-death-power` | float | `1.2` | Показатель степени в возрастной функции смертности |
| `--food-need-size-coef` | float | `5.0` | Вклад признака `size` в потребность в пище |
| `--food-need-resilience-coef` | float | `2.0` | Вклад признака `resilience` в потребность в пище |
| `--food-need-speed-coef` | float | `1.0` | Вклад признака `speed` в потребность в пище |
| `--food-need-aggr-coef` | float | `10.0` | Вклад признака `aggressiveness` в потребность в пище |

### Справка

```bash
python main.py --help
```

---

## Примеры

Полный прогон `Engine` (baseline + neural) на 600 шагов:

```bash
python main.py
```

Прогон с увеличенной силой нейросдвига и фиксированным сидом:

```bash
python main.py --shift-strength 0.9 --seed 7
```

Одиночный baseline-прогон с печатью метрик модели:

```bash
python synthetic_run.py --n-steps 200 --model-info
```

Сбор обучающей выборки для нейросоветчика:

```bash
python synthetic_run.py --n-steps 1000 --record --output data/run_01.json
```

Полный дебаг-прогон с пошаговыми логами:

```bash
python synthetic_run.py --n-steps 50 --debug --population-info --individual-info
```

Базовая визуализация (Plotly-графики в `vis/graphics/`):

```bash
python vis/vis_demo.py
```

---

## Сценарии

Готовые пресеты начальных условий и климатической динамики для запуска
симуляции в типовых режимах. Определены в `config/scenarios.py` —
каждый сценарий заменяет только `environment` в `SimConfig`, остальные
настройки (`population`, `individual`, `n_steps`, `seed` и др.) можно
переопределить через CLI.

### Доступные сценарии

| Имя             | Что моделирует                                              | Что нагружает              |
|-----------------|-------------------------------------------------------------|----------------------------|
| `stable`        | Статичная среда, без климатического давления                | Контрольный baseline       |
| `warming`       | Длительное монотонное потепление с холодного старта         | `heat_resistance`          |
| `ice_age`       | Похолодание + сжатие пищевой базы                           | `cold_resistance`, метаболизм |
| `harsh_seasons` | Быстрые тепловые циклы (~150 шагов на цикл)                 | Двусторонняя терморегуляция |
| `famine`        | Дефицит еды (≈4× ниже нормы), убывающая пищевая база        | Конкуренция, `size`, метаболизм |
| `hazardous`     | Уровень опасности растёт в 5× быстрее, старт выше           | `resilience`               |
| `chaos`         | Сезонность + растущая опасность + дефицит еды одновременно  | Стресс-тест адвайзера      |

---

## Профилирование

Профайлер прогоняет симуляцию под `cProfile` и собирает три артефакта в каталоге `reports/`:

| Файл | Назначение |
|---|---|
| `evosim.prof` | Бинарный дамп `cProfile` |
| `cprofile_summary.txt` | Текстовая сводка top-N по cumtime + tottime + callers |
| `profiling_report.html` | HTML-отчёт по коду проекта |

Импорты `mesa` / `numpy` / `pandas` / `scipy` в профилирование не включены.

### Запуск

```bash
python profile_run.py [опции]
```

Профайлер использует тот же CLI-парсер, поэтому все флаги симуляции (`--n-steps`, `--seed`, `--population-size`, `--mutation-std` и т.д.) применяются к прогоняемой модели как обычно. Дополнительные флаги, специфичные для профайлера:

| Флаг | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--output-dir` | str | `reports` | Каталог для всех артефактов |
| `--prof-filename` | str | `evosim.prof` | Имя `.prof` дампа |
| `--text-summary-filename` | str | `cprofile_summary.txt` | Имя текстовой сводки |
| `--html-report-filename` | str | `profiling_report.html` | Имя HTML-отчёта |
| `--top-n-text` | int | `50` | Сколько строк включать в текстовую сводку |
| `--top-n-callers` | int | `10` | Сколько горячих функций раскрывать в секции callers |
| `--top-n-flame` | int | `15` | Сколько строк показывать во флеймчарте HTML |
| `--top-n-console` | int | `15` | Сколько строк печатать в stdout по завершении |
| `--view` | flag | off | Автоматически открыть HTML-отчёт в браузере |

### Примеры

Базовый прогон, 200 шагов:

```bash
python profile_run.py --n-steps 200
```

С автооткрытием HTML-отчёта по завершении:

```bash
python profile_run.py --n-steps 500 --view
```

Углублённый анализ — больше callers и больше строк во флеймчарте:

```bash
python profile_run.py --n-steps 500 --top-n-callers 20 --top-n-flame 25
```

Профилирование с увеличенным стартовым размером популяции:

```bash
python profile_run.py --n-steps 1000 --population-size 300
```
```
