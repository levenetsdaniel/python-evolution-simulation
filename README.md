# Agent-based population evolution simulator with neural network guidance

## Пространственный режим

Основной режим `spatial` запускает двумерный мир без GUI. Классический
режим доступен через `mode=classic`. Координаты непрерывные, границы непроходимые,
а все случайные события используют `seed`. Пример запуска:

```bash
python3 main.py n_steps=300 seed=42 spatial_csv=artifacts/spatial_history.csv \
  spatial.initial_population=100 spatial.max_population=500 \
  spatial.width=100 spatial.height=100
```

`SpatialWorld` наследует `mesa.Model`, `SpatialIndividual` наследует базовый
`Individual` (а через него `mesa.Agent`). Пространственные варианты
`Population` и `Environment` находятся в тех же модулях, что и классические:
они управляют локальным размножением, смертностью, источниками пищи и общими
параметрами среды. Общие
энергетические расчёты, требования к пище, проверка возможности размножения,
скрещивание и мутация используются повторно.
Агенты регистрируются в Mesa, размещаются в непериодическом
`mesa.space.ContinuousSpace`; модель использует генератор Mesa, её счётчик
времени и `DataCollector`. Как и в baseline, модель координирует цикл, агент
хранит геном и состояние, а политика выделена отдельно. В пространственном
режиме этапы шага выполняются по ID, чтобы конкуренция за пищу не зависела от
порядка перебора `AgentSet`.

`spatial_csv` должен указывать на существующий каталог. CSV содержит шаг,
численность, среднюю энергию и возраст, среднюю продолжительность жизни умерших,
рождения, смерти по причинам, fitness, температуру, опасность, общий запас пищи,
число источников и события их смены, а также
средние значения каждого наследуемого признака.

Для интерактивной визуализации добавьте `view=true`:

```bash
python3 main.py view=true n_steps=200 seed=42 \
  spatial_view_interval=5 spatial_html=artifacts/plots/spatial/world.html
```

Откройте созданный HTML-файл в браузере. На карте показаны организмы (цвет —
энергия) и источники пищи (цвет и размер — доля заполнения). Под картой —
численность и общий запас пищи. Ползунок выбирает шаг, кнопки запускают и
останавливают анимацию. При наведении на организм показан fitness, в заголовке
шага — температура и опасность. `spatial_view_interval` задаёт частоту сохранения
кадров, но симуляция всё равно рассчитывает каждый шаг; нулевой и последний
шаги сохраняются всегда. HTML работает без сервера. Можно одновременно задать
`spatial_csv` для экспорта статистики. Начальные значения питания и локального
размножения подобраны для длительных прогонов; формула возрастной смертности
осталась общей с классическим режимом.

Из Python:

```python
from core.spatial import SpatialWorld

world = SpatialWorld(seed=42)
world.run(300)
world.export_csv("artifacts/spatial_history.csv")
```

Все параметры находятся в `SpatialConfig` (`config/sim_config.py`) и могут
переопределяться в Hydra через `spatial.<имя>=<значение>`. Начальные значения:
мир 100×100, 100 организмов, энергия 100, лимит 500, пять кластеров по двенадцать
источников. У каждого источника есть запас, ёмкость, радиус взаимодействия и
восстановление за шаг. После полного истощения источник исчезает с вероятностью
`spatial.food_disappearance_probability` (по умолчанию 0.6). Каждый свободный
слот затем даёт шанс `spatial.food_spawn_probability` (0.3) создать новый
источник в случайном месте мира. Начальные источники образуют кластеры, новые
распределяются по всей карте. Оба события используют `seed` и сохраняются в CSV.
Пример гарантированной смены полностью истощённых источников:
`spatial.food_disappearance_probability=1 spatial.food_spawn_probability=1`.
В пределах шага каждый организм выбирает ближайший доступный источник. У каждого
источника пища выдаётся в порядке ID с детерминированным сдвигом по шагу и
индексу источника, не больше `food_bite` и наследуемой
потребности `food_need`. Если запас кончился, оставшиеся рядом организмы вызывают
общий `Population.compete` и могут отнять пищу у получивших её **у того же
источника**. Энергия за пищу начисляется после столкновений, поэтому выигранная
пища питает победителя. Расход источника не превышает его запас; кража лишь
передаёт уже выданную пищу. Конфликт может расходовать энергию и привести к
смерти. Вероятности и стоимость атаки задаются через `population.attack_*` и
`population.wound_base`; CSV содержит `Deaths_competition`, `Attacks*`,
`FoodStolen`, `AttackEnergySpent`.

Скорость зависит от гена `speed`, дальность восприятия — от `resilience`.
Базовый расход, добавки за `speed` и `aggressiveness`, а также формула
возрастной смертности берутся из общего `Individual`. Коэффициенты задаются
через `individual.*`; дополнительно движение оплачивается по фактически
пройденному расстоянию (`spatial.movement_energy_cost`). После питания запас
ограничен `spatial.max_energy`. Организм погибает от истощения, низкого fitness
либо по возрасту.
Порог начальной и максимальной энергии задаётся через `spatial.initial_energy`
и `spatial.max_energy`; они переопределяют соответствующие значения
`IndividualConfig` только в пространственном режиме. Пища использует свой
коэффициент `spatial.food_energy`, поскольку пространственные источники
измеряются в отдельных единицах ресурса.

Как и в baseline, температура и опасность каждый шаг меняются по настройкам
`environment.*`. После движения и питания организм получает fitness из общей
функции: учитываются устойчивость к температуре, запас энергии, устойчивость
к опасности и соотношение признаков. Порог и вероятность смерти от fitness
задаются через `individual.fitness_death_threshold` и
`individual.fitness_death_prob_coef`; результаты пишутся в CSV как `AvgFitness`,
`Deaths_fitness` и `Deaths_threshold`. Пока условия одинаковы в любой точке
карты. Метод `environment.conditions_at(x, y)` оставляет место для будущих
регионов. Пример с классическим сценарием потепления:

```bash
python3 main.py environment=warming n_steps=300 spatial_csv=artifacts/spatial_history.csv
```

Поля `environment.food_availability` и `environment.food_regeneration_rate`
здесь не используются: количество и восстановление пищи задаются через `spatial.*`.

Размножение локальное: достаточно взрослая самка с нужным запасом энергии
может произвести потомка с самцом в радиусе `spatial.mating_radius=2` ячейки
локальной карты. Это евклидово расстояние между центрами организмов: при
`local_map_span=96` и `local_map_size=64` две ячейки равны 3 единицам мира.
Предел вычисляется как `mating_radius × local_map_span / local_map_size`,
так что при изменении масштаба карты меняется и физическое расстояние.
Партнёров организм замечает в пределах своего обычного радиуса восприятия;
рождение возможно только после сближения. Вероятность, как в baseline,
зависит от её насыщения.
Скрещивание и мутация используют те же операторы, что классический режим;
энергия потомка вычитается из энергии первого родителя. Рождение ограничено
`max_population`. Возраст потомка начинается с нуля. Самцы среди ближайших
кандидатов выбираются с весами по fitness, как в baseline. Мутация по умолчанию
остаётся случайной (`spatial.mutation_std`): отбор по fitness может сдвигать
средние признаки популяции, но мутация сама не знает «нужного» направления.
Для собственной направленной стратегии можно передать `mutation_strategy` в
`SpatialWorld`. CatBoost обучается отдельной командой до запуска симуляции:

```bash
python3 train_advisor.py simulation.mode=spatial simulation.n_steps=100 \
  simulation.seed=42 training_runs=1 iterations=100 \
  save_path=artifacts/models/spatial_advisor.cbm
```

Команда собирает рождения и первый измеренный fitness потомков в `TrainingBuffer`
из baseline-прогонов с обычной мутацией (в spatial — с эвристикой движения).
Признаки обучения и веса примеров остаются прежними: вход — геном до мутации,
среда и её изменение, средние признаки популяции и fitness родителей; цель —
фактический геном потомка, вес — улучшение его fitness относительно родителей.
При `training_runs > 1` используются последовательные seed. Если оценённых
потомков нет, команда сообщает об этом вместо сохранения необученного регрессора.
Рядом с `.cbm` сохраняется отчёт `.cbm.json` с режимом, seed и числом примеров.

Основная симуляция подключает готовый регрессор и использует фиксированные веса:

```bash
python3 main.py spatial_guidance.enabled=true n_steps=300 seed=42 \
  spatial_guidance.advisor_path=artifacts/models/spatial_advisor.cbm \
  spatial_guidance.shift_strength=0.7
```

Если файла нет, запуск завершается ошибкой; обучение автоматически не начинается.
В новых `.cbm` записаны режим симуляции, порядок генов и диапазон нормировки
температуры: загрузчик проверяет их совместимость. Старые `.cbm` без этих
метаданных поддерживаются с проверкой размерностей, но порядок генов необходимо
сверять самостоятельно. Пища нормируется на суммарную ёмкость источников мира.
Настроек `spatial_guidance.training_steps`, `iterations`, `retrain_steps` и
`save_path` больше нет; настройки обучения находятся в `config/advisor_config.yaml`.

Для классического сравнения используется тот же отдельный этап обучения:

```bash
python3 train_advisor.py simulation.mode=classic simulation.n_steps=100 \
  save_path=artifacts/models/classic_advisor.cbm
python3 comparison_run.py advisor_path=artifacts/models/classic_advisor.cbm
```

Обе ветви comparison запускаются без обучения и сбора обучающего буфера.
Для другого сценария обучения используйте
`environment@simulation.environment=warming` в команде `train_advisor.py`.

Классический запуск и эксперименты с CatBoost остаются доступны через
`python3 main.py mode=classic` и прежние команды сравнения.

Вызов `world.step(actions)` принимает словарь `id: MoveAction(dx, dy)`.
Непереданные действия выбирает `HeuristicPolicy` по наблюдениям
`SpatialObservationProvider`. Учитель сначала удовлетворяет голод, затем ищет
партнёра при готовности к размножению. При запасе энергии не выше 60%
(`spatial.foraging_threshold`) организм начинает искать пищу и продолжает до
85% (`spatial.satiation_threshold`); между порогами сохраняется предыдущее
состояние голода. Эта память относится к состоянию особи и не наследуется.
Организм движется к центру источника и может питаться по пути в радиусе потребления.
Сытый зрелый организм приближается к ближайшему видимому подходящему партнёру
и ждёт в пределах двух ячеек. При равном расстоянии выбирается меньший ID.
Если партнёр не виден, сытый зрелый организм направляется к ближайшему источнику
пищи как к месту встречи (`rendezvous`). Так организмы не расходятся сразу после
питания, что особенно важно при малом радиусе размножения. Если нужных целей
поблизости нет, он исследует пространство на своей скорости.
Сытые незрелые организмы и сытые организмы при заполненной популяции отдыхают.
Выбор родителей и рождение выполняет `Population`, а не политика движения.
Сначала обновляются общие условия и
вычисляются все действия, затем идут движение, энергозатраты, питание,
оценка fitness и смерть,
размножение, восстановление пищи и запись статистики. Обход организмов всегда
ведётся по ID; при равном расстоянии выбирается источник с меньшим индексом.
Физика мира не зависит от частоты визуализации. Наблюдения и политика выделены
отдельно для будущего подключения других источников наблюдений и политик.

Каждый организм предоставляет `organism.local_map`; та же карта доступна как
`world.observation_provider.observe(world, organism).local_map`. Это свежий
снимок состояния, массив NumPy `float32` формы `(5, 64, 64)` со значениями `[0, 1]`.
Каналы перечислены в `core.spatial_policy.LOCAL_MAP_CHANNELS`: пища (доля запаса
источника), другие живые организмы (занятость ячейки), граница (ячейки за пределами
мира), видимость, подходящие партнёры. Последний канал отмечает видимых живых
организмов противоположного пола, удовлетворяющих условиям размножения.
При попадании нескольких источников в ячейку берётся максимум
доли запаса. Сам наблюдатель в канале организмов не отображается. Объекты обозначены
точками, размер источника на карте не обозначает его радиус потребления.

Окно центрировано на организме и по умолчанию покрывает 96×96 единиц мира:
`spatial.local_map_span=96`, `spatial.local_map_size=64`. Столбцы растут по +x,
строки по +y; поворот организма не вращает карту. Видны центры ячеек в пределах
наследуемого радиуса восприятия; невидимые ячейки обнулены во всех каналах.
Маска отличает невидимое место от видимого пустого места. Физический масштаб
одинаков для всех организмов, объекты за пределами окна не попадают на карту.
Снимок защищён от записи и не изменяется после движения или рождения других
организмов; следующий запрос возвращает актуальную карту. Получение карты не
использует RNG и не зависит от HTML-визуализации. Для радиуса 32–48 можно задать
`spatial.base_perception=32 spatial.perception_trait_scale=16`.

### CNN для выбора цели

CNN выбирает тип поведения и конечную цель на видимых картах. Движение к
выбранной клетке рассчитывает независимый `TargetController` из `core`.
Установите PyTorch в окружении Python 3.13 и запустите обучение:

```bash
python3.13 -m pip install -r requirements.txt
python3.13 main.py spatial_cnn.enabled=true n_steps=200 seed=42 view=true \
  spatial_cnn.save_path=artifacts/models/spatial_target_policy.pt \
  spatial_html=artifacts/plots/spatial/cnn_target_world.html
```

Вход сети — две карты по пять каналов и 13 собственных признаков: энергия,
синус и косинус направления, семь генетических признаков, пол, готовность
к размножению и состояние голода. Обзорная карта имеет охват 96×96,
ближняя — `spatial.nearby_map_span=12.0`, обе по умолчанию имеют разрешение
`spatial.local_map_size=64`. Карты учитывают восприятие, границы и пригодность
партнёров. Абсолютные координаты и точные положения объектов из мира при
выполнении политики не используются.

**Выходы сети:**

- Тип цели: `forage`, `mate`, `rendezvous`, `explore`, `rest`.
- Карты оценок пищи и партнёров на обоих масштабах. Выбирается клетка с
  максимальной оценкой, а не среднее положение нескольких целей.

Кандидатами могут быть только клетки с видимыми объектами соответствующего
типа. Размножение требует собственной готовности; цели без видимых объектов
маскируются. Для оценки клеток используются общие свёртки и координатные
каналы центров пикселей относительно организма. Эти каналы задают геометрию
изображения; они не содержат координат объектов из состояния мира.
`CNNPolicy.select_targets(observations)` позволяет получить выбранные
`SpatialTarget` отдельно от выполнения движения.

**Исполнение:** центр клетки переводится в относительное положение цели.
`TargetController` ограничивает перемещение скоростью и замедляется при
приближении. У пищи он останавливается в пределах погрешности клетки;
для партнёра оставляет запас внутри радиуса размножения с учётом половины
диагонали клетки. `rest` даёт нулевое движение, `explore` — поворот и движение
с доступной скоростью. Мир применяет границы, затраты энергии, питание,
смертность и размножение. Цель пересматривается на каждом шаге, чтобы
учитывать движение партнёров и изменение ресурсов.

Регулятор одинаков для всех выбранных целей и не обращается к истинным
положениям пищи или партнёров. Прежняя подмена ближнего движения по точным
координатам убрана; настройки `precise_mating` и `mating_loss_weight` заменены
новой постановкой задачи выбора цели.

**Обучение:** перед основным прогоном эвристика работает в отдельных мирах.
Она размечает тип поведения и клетку ближайшей видимой цели, с предпочтением
ближней карты. Точные координаты учителя используются только для разметки.
Если его цель не представлена в доступных пикселях, пример получает режим
исследования без фиктивной клетки цели. Для отдыха и исследования пространственная
цель отсутствует. Обучение минимизирует кросс-энтропию типа поведения и клетки
цели (`spatial_cnn.target_loss_weight=1.0`). Типы поведения взвешиваются по
частотам сохранённых примеров. Карты, направления и метки клеток согласованно
поворачиваются на 0°, 90°, 180° и 270°.

По умолчанию: два мира по 80 шагов (`training_runs`, `training_steps`), до
4096 примеров (`max_samples`), 25 эпох (`epochs`), `batch_size=64`,
`learning_rate=0.0003`, `threads=1`. Буфер использует отдельный RNG;
запись наблюдений не меняет траекторию учителя. `n_steps` задаёт длину
основного прогона после обучения. В течение него веса фиксированы.

Проверка использует отдельный seed (`validation_steps=20`). Сохраняются веса
эпохи с минимальной проверочной функцией потерь. JSON рядом с моделью содержит
`validation_loss`, `goal_accuracy`, `target_accuracy`, результаты по типам
поведения и историю эпох. Точность клетки считается для размеченного типа
цели; её нужно рассматривать вместе с точностью типа поведения. Эти метрики
не заменяют проверки выживаемости в самостоятельных прогонах.

Загрузка готовой модели без повторного обучения:

```bash
python3.13 main.py spatial_cnn.enabled=true \
  spatial_cnn.model_path=artifacts/models/spatial_target_policy.pt \
  n_steps=300 seed=42 view=true \
  spatial_html=artifacts/plots/spatial/cnn_target_world.html
```

Формат модели — версия 4. Веса прежних версий движения несовместимы и требуют
переобучения. Проверяются геометрия карт, радиус размножения, порядок каналов
и генетических признаков. CNN можно совместить с заранее обученным CatBoost
через `spatial_guidance.enabled=true`. Эвристика доступна без установки PyTorch.

Текущие ограничения: нет регионов с разным климатом и хищничества. CNN обучается
подражать учителю питания, поиска партнёров и исследования; RL и оптимизации
пожизненного числа потомков пока нет. Подражание не гарантирует улучшения выживаемости,
особенно когда CNN уводит мир в состояния, отсутствовавшие в обучающих примерах.
Случайные повороты эвристики не предсказуемы из наблюдения, а детали положения
пищи теряются при растеризации. Пространственный guided запуск пока не выполняет
параллельного сравнения с baseline.
Шаги дискретны по времени;
скорость задаёт максимальное перемещение за один шаг.

---

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
├── artifacts/               # все output-артефакты: plots / dashboards / profiling
├── config/
│   ├── sim_config.py        # dataclass-схемы (Sim / Comparison / Profiler + подконфиги)
│   ├── output_paths.py      # единая схема путей для всех артефактов
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
- **2.4** Отдельное обучение советчика на baseline-данных; готовый регрессор подключается к симуляции с фиксированными весами
- **2.5** Сравнительный дашборд Plotly — анимация по поколениям: точки особей в пространстве признаков
- **2.6** Итоговая таблица baseline vs neural — численность, средний fitness, скорость адаптации

---

## Сборка

```bash
git clone https://github.com/levenetsdaniel/python-evolution-simulation
cd evosim

python3.13 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Для запуска через Hydra 1.3 используйте Python 3.13: в Python 3.14
`hydra-core` 1.3 вызывает ошибку `LazyCompletionHelp` ещё до старта симуляции.
Если активна Conda с Python 3.14, создайте окружение командой выше и запускайте
`main.py` после `source .venv/bin/activate`.

### Точки входа

| Скрипт | Что делает |
|---|---|
| `python3 main.py` | Основной прогон чистой симуляции. С `view=true` сохраняет графики состояния среды, энергии, демографии, причин смертей, прироста популяции и атак по одной ветке; с `model_info`, `population_info`, `individual_info` печатает диагностические сводки. |
| `python3 train_advisor.py` | Отдельное обучение CatBoost на baseline-данных spatial или classic; сохраняет `.cbm` и отчёт. |
| `python3 comparison_run.py` | Загружает готовый классический регрессор и запускает baseline + neural-guided с фиксированными весами. С `view=true` собирает сравнительный дашборд. |
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
python3 main.py environment.food_regeneration_rate=0.25
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
| `collect_agent_data` | bool | `false` | Сохранять per-agent историю `Fitness` / `Age` / `Energy`; включается автоматически при `individual_info=true` |

### `ComparisonConfig`

Используется только в `comparison_run.py` и для comparison dashboard.

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `simulation` | `SimConfig` | дефолты | Базовый конфиг обеих веток comparison pipeline |
| `advisor_path` | str | `artifacts/models/classic_advisor.cbm` | Готовый регрессор для классического сравнения |
| `shift_strength` | float | `0.7` | Сила сдвига генома в guided mutation |
| `x_trait` | str | `heat_resistance` | Ось X для animated scatter comparison dashboard |
| `y_trait` | str | `cold_resistance` | Ось Y для animated scatter comparison dashboard |
| `view` | bool | `false` | В `comparison_run.py` — собрать и сохранить comparison dashboard вместо обычного прогона |

### `environment` (`EnvironmentConfig`)

| Ключ | Тип | По умолчанию | Описание |
|---|---|---|---|
| `food_availability` | float | `10000.0` | Ёмкость пищевой базы |
| `food_regeneration_rate` | float | `1.0` | Доля дефицита пищи, восстанавливаемая за шаг; `1.0` полностью восполняет ресурс |
| `temp_start` | float | `20.0` | Стартовая температура (°C) |
| `temp_step` | float | `0.05` | Изменение температуры за шаг; на границах диапазона направление меняется плавно |
| `hazard_level_start` | float | `0.1` | Стартовый уровень опасности |
| `hazard_step` | float | `0.001` | Изменение опасности за шаг; на границах `[0, 1]` направление меняется плавно |
| `min_temperature` | float | `-30.0` | Нижняя граница температуры |
| `max_temperature` | float | `50.0` | Верхняя граница температуры |

### `population` (`PopulationConfig`)

| Ключ | Тип | По умолчанию | Описание                                              |
|---|---|---|-------------------------------------------------------|
| `initial_size` | int | `100` | Начальный размер популяции                            |
| `mutation_std` | float | `0.12` | Стандартное отклонение baseline-мутации               |
| `reproduction_rate` | float | `0.4` | Вероятность рождения потомка самкой с полным запасом энергии за шаг |
| `min_reproduction_age` | int | `2` | Минимальный возраст для размножения                   |
| `wound_base` | float | `0.4` | Базовая вероятность смерти проигравшего в конкуренции |
| `attack_base_probability` | float | `0.05` | Базовая готовность голодной особи вступить в конфликт |
| `attack_risk_aversion` | float | `0.6` | Насколько риск проигрыша снижает вероятность атаки |
| `attack_energy_cost` | float | `2.0` | Энергия, расходуемая на одну начатую атаку |
| `reproduction_energy_cost` | float | `15.0` | Энергия, расходуемая матерью при рождении потомка |
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
| `energy_capacity` | float | `100.0` | Максимальный запас энергии особи |
| `energy_initial` | float | `70.0` | Запас энергии при рождении или инициализации |
| `food_energy_conversion` | float | `0.5` | Сколько энергии даёт единица съеденной пищи |
| `basal_energy_cost` | float | `1.0` | Базовый расход энергии за шаг |
| `speed_energy_cost` | float | `2.0` | Дополнительный расход энергии от скорости |
| `aggressiveness_energy_cost` | float | `3.0` | Дополнительный расход энергии от агрессивности |

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
| `warming` | Потепление с холодного старта и слегка замедленным восстановлением пищи | `heat_resistance` |
| `ice_age` | Похолодание, меньшая ёмкость и медленное восстановление пищи | `cold_resistance`, метаболизм |
| `harsh_seasons` | Быстрые тепловые циклы и неполное восстановление ресурса | Двусторонняя терморегуляция |
| `famine` | Ёмкость пищи ≈4× ниже нормы и регенерация 15% дефицита за шаг | Конкуренция, `size`, метаболизм |
| `hazardous` | Уровень опасности растёт в 5× быстрее, восстановление пищи замедлено | `resilience` |
| `chaos` | Сезонность, растущая опасность, малая ёмкость и регенерация 30% | Стресс-тест |

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
python3 comparison_run.py advisor_path=artifacts/models/classic_advisor.cbm shift_strength=0.9
```

Полный дебаг-прогон с пошаговыми логами:

```bash
python3 main.py n_steps=50 debug=true population_info=true individual_info=true
```

Базовая визуализация (Plotly-графики в `artifacts/plots/demo/`):

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

Файл дашборда сохраняется в `artifacts/dashboards/comparison/dashboard.html`.

### Что внутри

- **Animated scatter (heat × cold).** Каждая точка — особь на двумерной проекции пространства признаков (по умолчанию `heat_resistance × cold_resistance`); цвет точки кодирует fitness (синяя гамма — baseline, красная — neural-guided). Внизу — кнопки play/pause и слайдер по поколениям для покадрового просмотра.
- **Comparison table.** Итоги двух ветвей по четырём метрикам:
  - финальная численность,
  - средний fitness в хвостовом окне (последние 50 шагов),
  - скорость адаптации — slope линейной аппроксимации `AvgFitness(time_step)`;
    это скорость на шаг симуляции, а не на биологическое поколение.
  - сколько шагов реально прожили (до экстинкции либо до `n_steps`).

  В каждой строке лучшее значение подсвечивается зелёным.

### Ось проекции

В animated scatter оси задаются через параметры `x_trait` и `y_trait` в `ComparisonConfig` (по умолчанию — `heat_resistance × cold_resistance`). Любое имя из `population.genome_labels` валидно:

```bash
python3 comparison_run.py view=true x_trait=size y_trait=speed
python3 comparison_run.py view=true simulation.environment=warming x_trait=heat_resistance y_trait=metabolic_rate
```

---

## Артефакты вывода

Все сценарии теперь складывают output в единый корень `artifacts/`:

| Каталог | Что появляется |
|---|---|
| `artifacts/plots/simulation/` | Plotly-графики из `python3 main.py view=true` |
| `artifacts/plots/demo/` | Plotly-графики из `python3 vis/vis_demo.py` |
| `artifacts/dashboards/comparison/` | `dashboard.html` из `python3 comparison_run.py view=true` |
| `artifacts/profiling/` | `.prof`, текстовая сводка и HTML-отчёт профайлера |
| `artifacts/hydra/` | служебные метаданные Hydra для одиночных и batch-запусков |

## Профилирование

Профайлер прогоняет симуляцию под `cProfile` и собирает три артефакта в каталоге `artifacts/profiling/`:

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
| `output_dir` | str | `artifacts/profiling` | Каталог для всех артефактов |
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
