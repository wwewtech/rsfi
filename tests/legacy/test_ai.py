import sys
import time
import numpy as np

from rsfi import RiemannianSphere, SphericalWhitening, RSFIFilter


def run_comprehensive_tests():
    print("=" * 80)
    print("      КОМПЛЕКСНЫЙ ТЕСТ МАТЕМАТИЧЕСКОЙ МОДЕЛИ СИСТЕМНОГО ДРЕЙФА И RSFI")
    print("=" * 80)

    dim = 1536  # Реальная размерность современных эмбеддингов (OpenAI / Llama)
    np.random.seed(2026)

    # --
    # ТЕСТ 1: ПРОВЕРКА АКСИОМ РИМАНОВОЙ ГЕОМЕТРИИ И ЛОГАРИФМИЧЕСКОГО ОПЕРАТОРА
    # --
    print("\n[ТЕСТ 1] Проверка аксиом Римановой геометрии на сфере S^(d-1)...")

    num_samples = 1000
    max_log_norm_error = 0.0
    max_tangent_ortho_error = 0.0

    S = RiemannianSphere.normalize(np.random.randn(dim))

    for _ in range(num_samples):
        R = RiemannianSphere.normalize(np.random.randn(dim))

        # Вычисление геодезической дистанции и логарифмического вектора
        d_geom = RiemannianSphere.geodesic_distance(S, R)
        v_R = RiemannianSphere.log_map(S, R)
        norm_v_R = np.linalg.norm(v_R)

        # Ошибка 1: ||Log_S(R)||_2 должно совпадать с d_M(S, R)
        log_norm_err = abs(norm_v_R - d_geom)
        if log_norm_err > max_log_norm_error:
            max_log_norm_error = log_norm_err

        # Ошибка 2: v_R должен быть строго ортогонален S (лежать в T_S M)
        tangent_err = abs(np.dot(S, v_R))
        if tangent_err > max_tangent_ortho_error:
            max_tangent_ortho_error = tangent_err

    print(f"  -> Макс. абсолютная невязка ||Log_S(R)||_2 - d_M(S, R): {max_log_norm_error:.2e}")
    print(f"  -> Макс. отклонение от ортогональности касательного пространства <S, v_R>: {max_tangent_ortho_error:.2e}")

    assert max_log_norm_error < 1e-12, "Ошибка геодезической нормы!"
    assert max_tangent_ortho_error < 1e-12, "Вектор не лежит в касательном пространстве!"
    print("  [РЕЗУЛЬТАТ]: ТЕСТ 1 УСПЕШНО ПРЕОДОЛЕН (Математическая строгость 100%)")

    # --
    # ТЕСТ 2: ТЕОРЕМА ПИФАГОРА И ОРТОГОНАЛЬНОЕ РАЗЛОЖЕНИЕ В T_S M
    # --
    print("\n[ТЕСТ 2] Строгая проверка ортогональности и Теоремы Пифагора в T_S M...")

    V_thr = RiemannianSphere.normalize(np.random.randn(dim))
    filter_sys = RSFIFilter(S, V_thr)

    max_pythagoras_error = 0.0
    max_vperp_ortho_error = 0.0

    for _ in range(num_samples):
        R = RiemannianSphere.normalize(np.random.randn(dim))
        res = filter_sys.evaluate(R)

        # 1. Невязка Теоремы Пифагора: ||v_R||^2 - (||v_perp||^2 + pi_thr^2)
        v_R_sq = res["d_M"] ** 2
        decomp_sq = res["norm_v_perp"] ** 2 + res["pi_thr"] ** 2
        pyth_err = abs(v_R_sq - decomp_sq)

        if pyth_err > max_pythagoras_error:
            max_pythagoras_error = pyth_err

        # 2. Ортогональность v_perp и e_thr: <v_perp, e_thr> должно быть 0
        vperp_err = abs(np.dot(res["v_perp"], filter_sys.e_thr))
        if vperp_err > max_vperp_ortho_error:
            max_vperp_ortho_error = vperp_err

    print(f"  -> Макс. абсолютная невязка Теоремы Пифагора в T_S M: {max_pythagoras_error:.2e}")
    print(f"  -> Макс. отклонение ортогональности компонентов <v_perp, e_thr>: {max_vperp_ortho_error:.2e}")

    assert max_pythagoras_error < 1e-12, "Нарушение теоремы Пифагора!"
    assert max_vperp_ortho_error < 1e-12, "Компоненты v_perp и e_thr не ортогональны!"
    print("  [РЕЗУЛЬТАТ]: ТЕСТ 2 УСПЕШНО ПРЕОДОЛЕН (Ортогональная развязка 100%)")


if __name__ == "__main__":
    run_comprehensive_tests()