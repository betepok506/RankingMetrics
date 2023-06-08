"""
Данный модуль содержит тесты для модуля `ranking_metrics.py`
"""
import unittest
from src.docs_ranking_metrics import RankingMetrics
from src.docs_ranking_metrics.evaluation_metrics import (
    TopK, AverageLoc, FDARO
)
from src.docs_ranking_metrics.ranking_metrics import (
    Bm25
)
import mock
from mock import Mock


class TestBm25(unittest.TestCase):
    def test_encode(self):
        bm25 = Bm25()
        tokenized_sentences = bm25._encode('t 87283 2sdfsd sfff')
        self.assertEqual(tokenized_sentences, [['t', '87283', '2sdfsd', 'sfff']])

        tokenized_sentences = bm25._encode(['aza baza', 'azaza baza first', '23 te1.'])
        self.assertEqual(tokenized_sentences, [['aza', 'baza'],
                                               ['azaza', 'baza', 'first'],
                                               ['23', 'te1.']])

    def test_sorted(self):
        bm25 = Bm25()
        sorting_result = bm25._sorted([0.87, 1, 0.6], [1, 0, 3])
        self.assertEqual(sorting_result, [(1, 0), (0.87, 1), (0.6, 3)])

        sorting_result = bm25._sorted([0.07, 0.1, 0.6], [0, 0, 3])
        self.assertEqual(sorting_result, [(0.6, 3), (0.1, 0), (0.07, 0)])

    def test_ranking(self):
        bm25 = Bm25()
        scores = bm25.ranking('windy London', ["Hello there good man!", "It is quite windy in London",
                                               "How is the weather today?"], [0, -1, 1])
        for ind, score in enumerate([(0.93729472, -1), (0., 0), (0., 1)]):
            self.assertAlmostEqual(scores[ind][0], score[0])


class TestRankingMetrics(unittest.TestCase):
    def test_init(self):
        mock_LaBSE = Mock()
        mock_LaBSE.name.return_value = "LaBSE"
        metrics = [mock_LaBSE]
        ranking_metrics = RankingMetrics(metrics)
        self.assertIsInstance(ranking_metrics.fake_top_k, TopK)
        self.assertIsInstance(ranking_metrics.average_place_fake_doc, AverageLoc)
        self.assertIsInstance(ranking_metrics.fake_doc_above_relevant_one, FDARO)

    def test_update(self):
        mock_LaBSE = Mock()
        mock_LaBSE.name.return_value = "LaBSE"
        mock_LaBSE.ranking.return_value = [(0.97, 2),
                                           (0.88, RankingMetrics.FAKE_DOC_LABEL),
                                           (0.76, 3),
                                           (0.001, 0)]

        mock_bm25 = Mock()
        mock_bm25.name.return_value = "Bm25"
        mock_bm25.ranking.return_value = [(0.97, 2),
                                          (0.88, RankingMetrics.FAKE_DOC_LABEL),
                                          (0.76, 3),
                                          (0.001, 0)]
        r_metrics = [mock_LaBSE, mock_bm25]
        metric = RankingMetrics(r_metrics, [1, 2])
        result = metric.get()
        for name_metric in result.keys():
            self.assertEqual(result[name_metric], 0)
        self.assertEqual(len(result), 16) # теперь считается 8 метрик для каждой модели

        metric.update('test_query', ['1', '2', '3', '4'], [1, -1, 2, 0])
        result = metric.get()
        self.assertEqual(len(result), 16)
        self.assertEqual(result['Bm25_Top@1'], 0)
        self.assertEqual(result['Bm25_Top@3'], 1)
        self.assertEqual(result['Bm25_Top@5'], 1)
        self.assertEqual(result['Bm25_FDARO@v1'], 0)
        self.assertEqual(result['Bm25_FDARO@v2'], 0)
        self.assertEqual(result['Bm25_AverageLoc'], 2)
        self.assertEqual(result['Bm25_AverageRelLoc'], 0.5)
        self.assertEqual(result['Bm25_UpQuartile'], 0)

        self.assertEqual(result['LaBSE_Top@1'], 0)
        self.assertEqual(result['LaBSE_Top@3'], 1)
        self.assertEqual(result['LaBSE_Top@5'], 1)
        self.assertEqual(result['LaBSE_FDARO@v1'], 0)
        self.assertEqual(result['LaBSE_FDARO@v2'], 0)
        self.assertEqual(result['LaBSE_AverageLoc'], 2)
        self.assertEqual(result['LaBSE_AverageRelLoc'], 0.5)
        self.assertEqual(result['LaBSE_UpQuartile'], 0)


if __name__ == "__main__":
    unittest.main()
