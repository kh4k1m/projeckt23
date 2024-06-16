# Сопроводительная документация
## Команда SKAI, UFA

### План документации
1. Иструкция по скачиванию образа,
2. Иструкция по запуску образа
3. Описание возможностей интерфейса
4. Описание решения

#### Скачивание образа
```sh
docker pull kh4k1m/project23:latest
```
#### Запуск образа
- Изменить /mnt/c/Users/salim/Projects/Project23/data/sample_videos на свой путь к папке с локальными данными
```sh
- docker run --gpus all --ipc=host --ulimit memlock=-1 -p 8501:8501 -it --ulimit stack=67108864 -v /mnt/c/Users/salim/Projects/Project23/data/sample_videos:/workspace/local_data kh4k1m/project23
``` 
- Войти по адресу в браузерной строке
```sh
http://localhost:8501/:8000
```
#### Описание решения
- Удаление дубликатов изображений для  предотвращения переусложнения и недообучения модели на этих дубликатах
  ![Процесс поиска дуюликатов](https://github.com/kh4k1m/projeckt23/blob/main/images/dublicates0.jpg)
  ![Процесс удаления дуюликатов](https://github.com/kh4k1m/projeckt23/blob/main/images/dublicates1.jpg)
- Удаление дубликатов “bboxes”
    - ![Пример дубликатов bboxes](https://github.com/kh4k1m/projeckt23/blob/main/images/dublicates2.jpg)
- Доразметка отсутствующих объектов: отсутствие данного этапа может привести к недостаточной обученности модели или ее низкой точности
    - ![Разметка объекта](https://github.com/kh4k1m/projeckt23/blob/main/images/misted.jpg)
- Получение дополнительных данных из датасета COCO по интересующим объектам (bird), так как после удаления дубликатов класса “bird” данных стало гораздо меньше
  - ![Пример дополнительных данных](https://github.com/kh4k1m/projeckt23/blob/main/images/bird1.jpg)
  - ![Пример дополнительных данных](https://github.com/kh4k1m/projeckt23/blob/main/images/bird.jpg)
- Разметка данных из открытых источников:
    - Kaggle.
    - RoboFlow
    - Youtube
    - ![Данные с COCO](https://github.com/kh4k1m/projeckt23/blob/main/images/copter1.jpg)
- Сборка своих беспилотных летательных аппаратов - коптера и самолета для изображений высокого качества классов “copter”, “bird”, “aircraft(4)”
- Метод кропа изображений вместо ресайза
    - ![Фича методы](https://github.com/kh4k1m/projeckt23/blob/main/images/cropped.jpg)
Идея заключается в том, чтобы не ресайзить изображения, так как объекты имеют  малый размер, а в том, чтобы поделить изображение на патчи 640x640 и обучать только на тех, которые содержат интересующий объект. Это позволяет использовать предобученные модели от YOLO, которые были обучены на разрешении 640x640
- Деление набора данных таким образом, чтобы домены валидации не пересекались с доменом тестовой и тренировочной выборки. Это позволяет понять обобщающие способности модели нейронных сетей. train, test, val: 14000, 3700, 2100 изображений соответственно.
- Модели YOLO:
    - YOLOv5 -  Хорошие способности обнаружения малых объектов из-за первого большЕго  сверточного слоя, который был изменен в модели V8.
    - YOLOv8 - Ускоренная модель, основанная на архитектуре “CSPNet” с “Anchor free” позволяет быстрее обнаруживать объекты.
    - YOLOv9 - Возможности архитектуры “GELAN” и метода “PGI” позволяют эффективно обнаруживать объекты разного масштаба.
- Использование технологий “Stacking” и “Bagging” для суммирования весов модели и возможности ансамблирования моделей.
- Подбор гиперпараметров эволюционными алгоритмами позволяют оптимизировать модель и повысить ее точность.
    - ![Результаты 1](https://github.com/kh4k1m/projeckt23/blob/main/images/result.jpg)
    - ![Результаты 2](https://github.com/kh4k1m/projeckt23/blob/main/images/result1.jpg)
Среди подготовленных моделей есть возможность выбрать быструю, точную и среднюю модели, в зависимости от задач и вычислительных возможностей
- Кастомная аугментация включает в себя отключение масштабирования изображения, чтобы обучаться на сжатых малых объектов, изменение цветовой палитры изображения и рандомные геометрические преобразования объекта

#### Описание интерфейса
-  Левая сторона окна позволяет изменять локальные настройки. 
-  Выбор модели (легкая, средняя, большая) - что позволяет значительно четче адаптировать наше решение под задачу:
    - Легкая модель (YOLOv8n) -  используется для решений, работающих в реальном времени для быстрого предсказания
    - Средняя модель (YOLOV9c) - универсальное решение, когда нет дополнительных условий усложнения.
    - Большая модель (YOLOv5x6) - высокая точность с большой задержкой. Модель используется для сложных объектов не в real-time 
- Проект позволяет выбирать выполнение модели как на графическом, так и на центральном процессоре
- Выбор уверенности модели помогает тонко настроить модель под свои нужды: так при наличие частых угроз уверенность стоит ставить ниже
- Программа позваляет сортировать объекты по типам и по клику выбирать нужные изображения или видеоряд

