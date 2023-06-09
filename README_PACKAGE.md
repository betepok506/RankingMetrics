# Описание
Данный репозиторий содержит реализацию алгоритмов ранжирования Bm25, LaBSE, MsMarcoST, MsMarcoCE 
с подсчетом метрик: 
* Top@1;
* Top@3;
* Top@5;
* Средняя позиция в выдачах (AverageLoc);
* Cредняя относительная позиция в выдачах (AverageRelLoc);
* Оценка как часто фейковый документ выше всех релевантных (FDARO@v1);
* Оценка как часто фейковый документ выше хотя бы одного релевантного (FDARO@v2);
* Частота попадания фейкового документа в топ 25% (UpQuartile).



# Установка
Для установки пакета воспользуйтесь командой
```
pip install docs-ranking-metrics
```

# Пример использования
Пример использования представлен в `examples/using_metrics.py`

```commandline
# Объявление метрик
metrics = [LaBSE(), Bm25()]
# Объявление класса агрегирующего обновление метрик
rank_metrics = RankingMetrics(metrics)

...

'''
Обновление значений метрик, где 
query - запрос по которому сгенерирован документ, 
sentences - массив документов,
labels - метки документов
'''
rank_metrics.update(query, sentences, labels)

...
# Получение значений подсчитанных метрик ввиде словаря
rank_metrics.get()
# Получение значений метрик при помощи функции show_metrics
rank_metrics.show_metrics()
```

Возможный вывод метода get:
```
{
    'LaBSE_AverageLoc': 10.5, 
    'Bm25_AverageLoc': 1.13513, 
    'LaBSE_Top@1': 0.0, 
    'LaBSE_Top@3': 0.013513, 
    'LaBSE_Top@5': 0.013513, 
    'Bm25_Top@1': 0.91891, 
    'Bm25_Top@3': 1.0, 
    'Bm25_Top@5': 1.0, 
    'LaBSE_FDARO': 0.6216, 
    'Bm25_FDARO': 1.0
}
```

Возможный вывод метода show_metrics():
```
LaBSE_AverageLoc: 4.5   Bm25_AverageLoc: 3.0   
-----------------------------
LaBSE_AverageRelLoc: 0.75   Bm25_AverageRelLoc: 0.5   
-----------------------------
LaBSE_Top@1: 0.0   Bm25_Top@1: 0.5   
LaBSE_Top@3: 0.5   Bm25_Top@3: 0.5   
LaBSE_Top@5: 0.5   Bm25_Top@5: 1.0   
-----------------------------
LaBSE_FDARO@v1: 0.5   Bm25_FDARO@v1: 0.5   
LaBSE_FDARO@v2: 0.5   Bm25_FDARO@v2: 0.5   
-----------------------------
LaBSE_UpQuartile: 0.5   Bm25_UpQuartile: 0.5 
```
