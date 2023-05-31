"""
Данный модуль содержит тесты для модуля `evaluation_metrics.py`
"""
import unittest
from src.docs_ranking_metrics import RankingMetrics, Bm25
from src.docs_ranking_metrics.evaluation_metrics import (
    TopK, AverageLoc, FDARO
)
import mock
from mock import Mock


class TestTopK(unittest.TestCase):
    def test_init(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"

        mock_labse = Mock()
        mock_labse.name.return_value = "LaBSE"

        r_metrics = [mock_bm25, mock_labse]
        metric = TopK(r_metrics)
        self.assertEqual(len(metric.metrics), len(metric._top_numbers) * len(r_metrics))
        self.assertEqual(len(metric.calls_cnt), len(metric._top_numbers) * len(r_metrics))
        cnt = 0
        for name_metric in metric.metrics.keys():
            if name_metric in metric.calls_cnt:
                cnt += 1

        # Сравнение ключей в обоих списках, они должны быть одинаковы
        self.assertEqual(len(metric.metrics), cnt)

        # Проверка что словари инициализированны нулями
        for name_metric in metric.metrics.keys():
            self.assertEqual(metric.metrics[name_metric], 0)
            self.assertEqual(metric.calls_cnt[name_metric], 0)

        # Проверка правильности генерации ключей в словаре
        for cur_metric in r_metrics:
            for cur_top in metric._top_numbers:
                self.assertIn(cur_metric.name() + metric._separator + metric.name() + str(cur_top),
                              metric.metrics)

    def test_update_case_1(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = TopK(r_metrics)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0), (0.43, 1)], 5.3)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [("0.87", 0), (0.43, 1)], [5, 3])

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0.8), (0.43, 1)], [5, 3])

    def test_update_case_2(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = TopK(r_metrics)

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 1)], RankingMetrics.FAKE_DOC_LABEL)
        self.assertEqual(metric.metrics["Bm25_Top@1"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@3"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@5"], 1)

        self.assertEqual(metric.calls_cnt["Bm25_Top@1"], 1)
        self.assertEqual(metric.calls_cnt["Bm25_Top@3"], 1)
        self.assertEqual(metric.calls_cnt["Bm25_Top@5"], 1)

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics["Bm25_Top@1"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@3"], 2)
        self.assertEqual(metric.metrics["Bm25_Top@5"], 2)

        self.assertEqual(metric.calls_cnt["Bm25_Top@1"], 2)
        self.assertEqual(metric.calls_cnt["Bm25_Top@3"], 2)
        self.assertEqual(metric.calls_cnt["Bm25_Top@5"], 2)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics["Bm25_Top@1"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@3"], 2)
        self.assertEqual(metric.metrics["Bm25_Top@5"], 3)

        self.assertEqual(metric.calls_cnt["Bm25_Top@1"], 3)
        self.assertEqual(metric.calls_cnt["Bm25_Top@3"], 3)
        self.assertEqual(metric.calls_cnt["Bm25_Top@5"], 3)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, RankingMetrics.FAKE_DOC_LABEL)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics["Bm25_Top@1"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@3"], 2)
        self.assertEqual(metric.metrics["Bm25_Top@5"], 3)

        self.assertEqual(metric.calls_cnt["Bm25_Top@1"], 4)
        self.assertEqual(metric.calls_cnt["Bm25_Top@3"], 4)
        self.assertEqual(metric.calls_cnt["Bm25_Top@5"], 4)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics["Bm25_Top@1"], 1)
        self.assertEqual(metric.metrics["Bm25_Top@3"], 2)
        self.assertEqual(metric.metrics["Bm25_Top@5"], 3)

        self.assertEqual(metric.calls_cnt["Bm25_Top@1"], 5)
        self.assertEqual(metric.calls_cnt["Bm25_Top@3"], 5)
        self.assertEqual(metric.calls_cnt["Bm25_Top@5"], 5)

    def test_get(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = TopK(r_metrics)

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 1)], RankingMetrics.FAKE_DOC_LABEL)
        result = metric.get()
        answer = {
            'Bm25_Top@1': 1,
            'Bm25_Top@3': 1,
            'Bm25_Top@5': 1,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            'Bm25_Top@1': 0.5,
            'Bm25_Top@3': 1,
            'Bm25_Top@5': 1,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            'Bm25_Top@1': 1 / 3,
            'Bm25_Top@3': 2 / 3,
            'Bm25_Top@5': 1,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, RankingMetrics.FAKE_DOC_LABEL)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            'Bm25_Top@1': 1 / 4,
            'Bm25_Top@3': 2 / 4,
            'Bm25_Top@5': 3 / 4,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            'Bm25_Top@1': 1 / 5,
            'Bm25_Top@3': 2 / 5,
            'Bm25_Top@5': 3 / 5,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])


class TestFDARO(unittest.TestCase):
    def test_init(self):
        vesrions = ["v1", "v2"]

        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"

        mock_labse = Mock()
        mock_labse.name.return_value = "LaBSE"

        r_metrics = [mock_bm25, mock_labse]
        metric = FDARO(r_metrics)
        # умножаем длину ответа на 2, так как две версии FDARO
        self.assertEqual(len(metric.metrics), len(r_metrics) * 2)
        self.assertEqual(len(metric.calls_cnt), len(r_metrics) * 2)
        cnt = 0
        for name_metric in metric.metrics.keys():
            if name_metric in metric.calls_cnt:
                cnt += 1

        # Сравнение ключей в обоих списках, они должны быть одинаковы
        self.assertEqual(len(metric.metrics), cnt)

        # Проверка что словари инициализированны нулями
        for name_metric in metric.metrics.keys():
            self.assertEqual(metric.metrics[name_metric], 0)
            self.assertEqual(metric.calls_cnt[name_metric], 0)

        # Проверка правильности генерации ключей в словаре
        for cur_metric in r_metrics:
            self.assertIn(cur_metric.name() + metric._separator + metric.name() + vesrions[0], metric.metrics)
            self.assertIn(cur_metric.name() + metric._separator + metric.name() + vesrions[1], metric.metrics)


    def test_update_case_1(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = FDARO(r_metrics)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0), (0.43, 1)], 5.3)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [("0.87", 0), (0.43, 1)], [5, 3])

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0.8), (0.43, 1)], [5, 3])

    def test_update_case_2(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = FDARO(r_metrics, [1, 2, 3])

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 2)], RankingMetrics.FAKE_DOC_LABEL)
        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}v1"], 1)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}v1"], 1)

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}v1"], 2)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}v1"], 2)

        metric.update("Bm25", [(0.88, 3), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}v1"], 2)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}v1"], 3)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 1), (0.21, RankingMetrics.FAKE_DOC_LABEL)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}v1"], 2)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}v1"], 4)

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 2),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}v1"], 3)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}v1"], 5)

    def test_get(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = FDARO(r_metrics, [1, 2, 3])

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 2)], RankingMetrics.FAKE_DOC_LABEL)
        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}v1": 1,
            f"Bm25_{metric.name()}v2": 1
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 2), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}v1": 1 / 2,
            f"Bm25_{metric.name()}v2": 1 / 2
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 2)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}v1": 2 / 3,
            f"Bm25_{metric.name()}v2": 2 / 3
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, RankingMetrics.FAKE_DOC_LABEL),
                               (0.43, 0), (0.22, 0), (0.21, 3)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}v1": 3 / 4,
            f"Bm25_{metric.name()}v2": 3 / 4,
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}v1": 3 / 5,
            f"Bm25_{metric.name()}v2": 3 / 5
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])


class TestAverageLoc(unittest.TestCase):
    def test_init(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"

        mock_labse = Mock()
        mock_labse.name.return_value = "LaBSE"

        r_metrics = [mock_bm25, mock_labse]
        metric = AverageLoc(r_metrics)
        self.assertEqual(len(metric.metrics), len(r_metrics))
        self.assertEqual(len(metric.calls_cnt), len(r_metrics))
        cnt = 0
        for name_metric in metric.metrics.keys():
            if name_metric in metric.calls_cnt:
                cnt += 1

        # Сравнение ключей в обоих списках, они должны быть одинаковы
        self.assertEqual(len(metric.metrics), cnt)

        # Проверка что словари инициализированны нулями
        for name_metric in metric.metrics.keys():
            self.assertEqual(metric.metrics[name_metric], 0)
            self.assertEqual(metric.calls_cnt[name_metric], 0)

        # Проверка правильности генерации ключей в словаре
        for cur_metric in r_metrics:
            self.assertIn(cur_metric.name() + metric._separator + metric.name(), metric.metrics)

    def test_update_case_1(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = AverageLoc(r_metrics)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0), (0.43, 1)], 5.3)

        with self.assertRaises(TypeError):
            metric.update("Bm25", [("0.87", 0), (0.43, 1)], [5, 3])

        with self.assertRaises(TypeError):
            metric.update("Bm25", [(0.87, 0.8), (0.43, 1)], [5, 3])

    def test_update_case_2(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = AverageLoc(r_metrics)

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 1)], RankingMetrics.FAKE_DOC_LABEL)
        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}"], 1)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}"], 1)

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}"], 3)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}"], 2)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}"], 8)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}"], 3)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, RankingMetrics.FAKE_DOC_LABEL)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}"], 14)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}"], 4)

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        self.assertEqual(metric.metrics[f"Bm25_{metric.name()}"], 21)
        self.assertEqual(metric.calls_cnt[f"Bm25_{metric.name()}"], 5)

    def test_get(self):
        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        r_metrics = [mock_bm25]
        metric = AverageLoc(r_metrics)

        metric.update("Bm25", [(0.88, RankingMetrics.FAKE_DOC_LABEL), (0.69, 1)], RankingMetrics.FAKE_DOC_LABEL)
        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}": 1
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, RankingMetrics.FAKE_DOC_LABEL), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}": 3 / 2
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, RankingMetrics.FAKE_DOC_LABEL), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}": (1 + 2 + 5) / 3
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, RankingMetrics.FAKE_DOC_LABEL)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}": (1 + 2 + 5 + 6) / 4
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])

        metric.update("Bm25", [(0.88, 0), (0.69, 0), (0.44, 0),
                               (0.43, 0), (0.22, 0), (0.21, 0)],
                      RankingMetrics.FAKE_DOC_LABEL)

        result = metric.get()
        answer = {
            f"Bm25_{metric.name()}": (1 + 2 + 5 + 6 + 7) / 5
        }
        self.assertEqual(len(result), len(answer))
        for name_metric in result.keys():
            self.assertAlmostEqual(result[name_metric], answer[name_metric])


if __name__ == "__main__":
    unittest.main()
