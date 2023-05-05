from src.ranking_metrics.ranking_metrics import (
    RankingMetrics,
    Bm25,
    LaBSE
)

data = [{"query": ")what was the immediate impact of the success of the manhattan project?",
         "passage_text": [
             "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.",
             "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.",
             "Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade."],
         "is_selected": [1, 0, 2]}]

if __name__ == "__main__":
    # Объявление метрик
    metrics = [Bm25(), LaBSE()]
    # Объявление класса агрегирующего обновление метрик
    rm = RankingMetrics(metrics)
    for item in data:
        '''
            Обновление значений метрик, где 
            query - запрос по которому сгенерирован документ, 
            sentences - массив документов,
            labels - метки документов
        '''
        rm.update(item["query"], item["passage_text"], item["is_selected"])
        print(rm.get())
