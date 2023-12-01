# Как использовать
Создаем виртуальное окружение для двух версий Python:
* Python3.10 - основное
* Python3.7 - только для библиотеки `neuralcoref`

Устанавливаем зависимости из файлов `requirements` в соответсвующие виртуальные окружения.

Затем запускаем сервер FastApi, к которому будет обращаться основная программма для решения задачи *coreference resolution*:

```
uvicorn coref_resolution:app --port 8001 --reload

```

Наконец, для решения задачи факчкинга необходимо создать объект класса Factchecker и в функцию `start_factchecking` передать текст, сгенерированный LLM. 






