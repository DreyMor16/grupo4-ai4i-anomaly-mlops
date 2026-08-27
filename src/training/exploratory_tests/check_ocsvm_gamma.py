import sys
from pathlib import Path

from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.feature_engineering.preprocessing import preprocesar_datos


RANDOM_STATE = 42


def main():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = preprocesar_datos(
        feature_set="engineered_only",
        approach="semi_supervised",
        random_state=RANDOM_STATE
    )

    modelo = OneClassSVM(
        nu=0.02,
        gamma="scale",
        kernel="rbf"
    )

    modelo.fit(
        X_train
    )

    print("\n==============================================")
    print("EXPLORACIÓN GAMMA ONE-CLASS SVM")
    print("==============================================")

    print(
        f"Gamma configurado: scale"
    )

    print(
        f"Gamma efectivo: {modelo._gamma}"
    )

    print(
        f"Número de features: {X_train.shape[1]}"
    )


if __name__ == "__main__":
    main()