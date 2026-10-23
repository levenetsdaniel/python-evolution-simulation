# EvoSim

Агентная симуляция эволюции популяции на Python и Mesa. Организмы наследуют семь признаков, расходуют энергию, конкурируют за пищу, размножаются и адаптируются к изменяющейся среде.

Поддерживаются два режима: **spatial** — непрерывный двумерный мир с локальными источниками пищи; **classic** — популяционная модель без пространственного перемещения. Режим по умолчанию — `spatial`.

## Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация и сценарии](#конфигурация-и-сценарии)
- [Машинное обучение](#машинное-обучение)
- [Тесты](#тесты)
- [Структура проекта](#структура-проекта)

## Установка

Для локального окружения используется Python **3.13**.

Из корня проекта:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Основные библиотеки: Mesa, NumPy, pandas, SciPy, Hydra/OmegaConf, Plotly, CatBoost и PyTorch. PyTorch нужен для CNN; полный `requirements.txt` устанавливает его вместе с остальными зависимостями. Lock-файл пока отсутствует.

## Быстрый старт

Обычная симуляция без обучения и готовых моделей:

```bash
python main.py n_steps=300 seed=42
```

HTML с картой организмов, источниками пищи и статистикой:

```bash
python main.py n_steps=200 seed=42 view=true
```

Результат: `artifacts/plots/spatial/world.html`. Откройте файл в браузере; сервер не нужен. По умолчанию сохраняется каждый шаг. Ползунок и поле «Шаг» выбирают кадр, «Шаг воспроизведения» задаёт интервал анимации. При наведении на организм показано его поле зрения.

Для уменьшения числа сохранённых кадров задайте `spatial_view_interval=5` — расчёт по-прежнему выполняет каждый шаг, а в HTML будут доступны сохранённые кадры. Нулевой и последний шаги сохраняются всегда.

Экспорт статистики пространственного режима в CSV:

```bash
python main.py n_steps=300 seed=42 spatial_csv=artifacts/spatial_history.csv
```

CSV содержит численность, энергию, рождения и причины смертей, fitness, параметры среды, запасы пищи и средние значения генов. Его можно сохранять одновременно с HTML через `view=true`.

Классический режим:

```bash
python main.py mode=classic n_steps=300 model_info=true
python main.py mode=classic n_steps=200 view=true
```

Графики классического режима сохраняются в `artifacts/plots/simulation/`.

## Конфигурация и сценарии

CLI использует Hydra: переопределения передаются как `ключ=значение`, вложенные параметры — через точку. Запускайте команды из корня проекта.

```bash
python main.py environment=famine n_steps=500 seed=45
python main.py spatial.initial_population=150 spatial.max_population=600
python main.py --cfg job
python main.py --help
```

| Группа | Назначение |
| --- | --- |
| `environment` | Температура, опасность, доступность и восстановление пищи |
| `population` | Геном и конкуренция; численность и размножение classic |
| `individual` | Метаболизм, потребность в пище, возрастная смертность |
| `fitness` | Коэффициенты приспособленности |
| `spatial` | Размер мира, численность spatial, движение, источники, локальное размножение |
| `spatial_guidance` | CatBoost для мутаций потомства в spatial |
| `spatial_cnn` | CNN для выбора поведения и пространственной цели |

Сценарии: `default`, `stable`, `warming`, `ice_age`, `harsh_seasons`, `famine`, `hazardous`, `chaos`. Их определения: [config/scenarios.py](config/scenarios.py).

Все сценарии работают в обоих режимах. В spatial ёмкость и начальный запас источника умножаются на `environment.food_availability / 10000`, а скорость восстановления — на `environment.food_regeneration_rate`. Например, `famine` при настройках по умолчанию даёт 25 единиц запаса и восстановление 1.5 единицы за шаг на источник. Новые источники получают те же параметры; число источников и радиус питания задаются через `spatial.*`.

В classic `food_regeneration_rate` задаёт долю недостающей пищи, восстанавливаемую за шаг. Температура и опасность в обоих режимах отражаются от заданных границ.

Значения по умолчанию и проверки параметров: [config/sim_config.py](config/sim_config.py).

| Точка входа | Конфигурация |
| --- | --- |
| `python main.py` | `config/sim_config.yaml` |
| `python -m neural.advisor_training` | `config/advisor_config.yaml` |
| `python comparison_run.py` | `config/compare_config.yaml` |

Hydra сохраняет параметры запуска и логи в `artifacts/hydra/`.

## Машинное обучение

CatBoost и CNN решают разные задачи и включаются отдельно. CatBoost — градиентный бустинг, который направляет мутации генома. CNN имитирует выбор поведения и цели; логика движения, питания и размножения остаётся в ядре.

### CatBoost: обучение и подключение

Обучение пространственного советчика на независимых baseline-прогонах:

```bash
python -m neural.advisor_training simulation.mode=spatial simulation.n_steps=300 \
  training_runs=2 validation_runs=1
python main.py spatial_guidance.enabled=true n_steps=300
```

Сбор данных идёт с обычной случайной мутацией. Для обучения на нескольких сценариях добавьте, например, `'training_scenarios=[stable,warming,ice_age,famine]'`. `training_runs` задаёт число seed на каждый сценарий, `validation_runs` — число отдельных проверочных seed. Параметры среды для одного прогона можно выбрать через `environment@simulation.environment=warming`.

Обучение классического советчика и сравнение двух ветвей:

```bash
python -m neural.advisor_training simulation.mode=classic simulation.n_steps=300
python comparison_run.py simulation.n_steps=300 view=true
```

Модели сохраняются в `artifacts/models/`: `spatial_advisor.cbm` и `classic_advisor.cbm`; рядом создаётся отчёт `.cbm.json`. Симуляция загружает готовый советчик с фиксированными весами. `spatial_guidance.advisor_path` задаёт путь в spatial, `advisor_path` — в сравнении classic.

Сравнительный дашборд: `artifacts/dashboards/comparison/dashboard.html`. Он загружает Plotly через CDN и требует доступа к сети. `comparison_run.py` поддерживает только `classic` и требует предварительно обученной модели.

### CNN: обучение и загрузка

Для обучения явно передайте пустой `model_path`:

```bash
python main.py spatial_cnn.enabled=true 'spatial_cnn.model_path=""' \
  spatial_cnn.device=auto n_steps=0
```

`spatial_cnn.training_steps` задаёт длительность каждого обучающего прогона учителя, `training_runs` — их количество. `n_steps` управляет последующей симуляцией; в примере она содержит ноль шагов. CNN учится на примерах выбора поведения и видимой целевой ячейки эвристикой, а выживаемость оценивается отдельными прогонами.

Обучение сохраняет веса в `artifacts/models/spatial_target_policy.pt` и отчёт `.pt.json`. Для другого файла задайте `spatial_cnn.save_path`. `device=auto` выбирает CUDA, затем Apple MPS, затем CPU; инференс выполняется на CPU.

Загрузка готовых весов:

```bash
python main.py spatial_cnn.enabled=true n_steps=200 view=true
```

При непустом `model_path` запускается загрузка; отсутствие файла приводит к ошибке. После свежего клонирования сначала обучите модель или получите совместимые веса отдельно.

По умолчанию `model_path=artifacts/models/spatial_target_policy.pt`, поэтому одного `spatial_cnn.enabled=true` достаточно для загрузки. Путь выводится как `CNN model loaded: ...`. Показанные при загрузке метрики взяты из сохранённого отчёта обучения.

Файл модели содержит геометрию карт, порядок каналов и генов, радиус размножения. Эти параметры проверяются при загрузке; для другой геометрии нужны соответствующие веса. CatBoost и CNN можно включить одновременно, добавив `spatial_guidance.enabled=true` к запуску готовой CNN.

## Тесты

```bash
python -m pytest -q
python tests_run.py
python -m pytest -q tests/spatial_tests.py
```

Обнаружение файлов `*_tests.py` настроено в `pytest.ini`. Набор проверяет ядро, конфигурации, точки входа, пространственное поведение, визуализацию и ML-компоненты.

## Структура проекта

Основные каталоги и точки входа:

```text
evosim/
├── config/                # dataclass-схемы, валидация, Hydra, сценарии
├── core/                  # модели, агенты, среда, генетика, fitness
├── neural/                # CatBoost, CNN, сбор данных и обучение
│   ├── advisor_training.py # отдельное обучение CatBoost: python -m neural.advisor_training
│   └── cnn_training.py     # сбор примеров, обучение и загрузка CNN через main.py
├── vis/                   # Plotly-графики, карта, сравнительный дашборд
├── tests/                 # тестовый набор pytest
├── artifacts/             # локальные результаты; исключены из Git
├── main.py                # пространственный или классический прогон
├── comparison_run.py      # classic: baseline против CatBoost
├── tests_run.py           # запуск тестов
├── pytest.ini
└── requirements.txt
```
