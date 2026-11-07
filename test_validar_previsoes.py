import unittest

import numpy as np
import pandas as pd

from validar_previsoes import (
    calcular_metricas,
    converter_para_base_precos,
    gerar_divisoes_temporais,
    previsoes_referencia,
)


class TestValidacaoPrevisoes(unittest.TestCase):
    def test_converte_valor_para_anos_anteriores_e_posteriores(self):
        inpc = pd.Series({2020: 10.0, 2021: 20.0, 2022: 5.0})

        self.assertAlmostEqual(
            converter_para_base_precos(100, 2020, 2022, inpc),
            126,
        )
        self.assertAlmostEqual(
            converter_para_base_precos(126, 2022, 2020, inpc),
            100,
        )

    def test_conversao_falha_se_inpc_necessario_nao_existe(self):
        with self.assertRaisesRegex(ValueError, "INPC ausente"):
            converter_para_base_precos(100, 2020, 2022, pd.Series({2021: 5.0}))

    def test_divisoes_temporais_preservam_ordem_e_janela_expansiva(self):
        divisoes = gerar_divisoes_temporais(range(2016, 2026), 5)

        self.assertEqual(len(divisoes), 5)
        np.testing.assert_array_equal(divisoes[0][0], range(2016, 2021))
        self.assertEqual(divisoes[0][1], 2021)
        np.testing.assert_array_equal(divisoes[-1][0], range(2016, 2025))
        self.assertEqual(divisoes[-1][1], 2025)

    def test_divisoes_rejeitam_anos_ausentes(self):
        with self.assertRaisesRegex(ValueError, "contínua"):
            gerar_divisoes_temporais([2016, 2017, 2019, 2020], 2)

    def test_previsoes_de_referencia(self):
        previsoes = previsoes_referencia(
            [2018, 2019, 2020],
            [1, 2, 3],
            2021,
        )

        self.assertEqual(previsoes["Último valor"], 3)
        self.assertEqual(previsoes["Média histórica"], 2)
        self.assertAlmostEqual(previsoes["Tendência linear"], 4)

    def test_metricas_incluem_erros_e_cobertura_do_intervalo(self):
        previsoes = pd.DataFrame(
            {
                "previsao_media": [10.0, 12.0, 8.0],
                "observado": [9.0, 10.0, 10.0],
                "p10": [8.0, 11.0, np.nan],
                "p90": [11.0, 13.0, np.nan],
            }
        )

        metricas = calcular_metricas(previsoes)

        self.assertEqual(metricas["n_previsoes"], 3)
        self.assertAlmostEqual(metricas["mae"], 5 / 3)
        self.assertAlmostEqual(metricas["rmse"], np.sqrt(3))
        self.assertAlmostEqual(metricas["vies_medio"], 1 / 3)
        self.assertEqual(metricas["cobertura_80_pct"], 50)
        self.assertEqual(metricas["largura_media_intervalo"], 2.5)

    def test_metricas_sem_intervalo_nao_inventam_cobertura(self):
        previsoes = pd.DataFrame(
            {
                "previsao_media": [1.0, 3.0],
                "observado": [2.0, 1.0],
                "p10": [np.nan, np.nan],
                "p90": [np.nan, np.nan],
            }
        )

        metricas = calcular_metricas(previsoes)

        self.assertTrue(np.isnan(metricas["cobertura_80_pct"]))
        self.assertTrue(np.isnan(metricas["largura_media_intervalo"]))


if __name__ == "__main__":
    unittest.main()
